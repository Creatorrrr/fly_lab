# M3 S16: 실제 JSON negative 오류와 immutable proof 복제 비용 수정 설계

2026-10-09 ROOT 설계 초안. 목표는 기존 full4 검증을 동일한180/160/20 안에서 통과시켜 실제 BANC 보행 후보 검증을 다시 진행하는 것이다. full4 통과는 F1–F8 완료나 자연스러운 생물학적 보행 증명이 아니다. 사용자의 최신 자율 진행 지시에 따라 추가 승인 질문 없이 설계·Astra 리뷰·Sol 구현·ROOT 리뷰를 수행하되, 기존 전체17900초와 모든 누적 비용·과학적 기준을 유지한다.

## 실제 근거

B는 verification/banc-natural-gait-m3-20260919이다. S15 live freeze B/s15-s1.json262138/2af47fcbf595e8e89d0b67f1e35d56fe57f967172c70a8f4ef01d7c8c0d5c29c, review246946/7d8d8039636608b0289b023cee499b0b35451568dc979c04d70f2c8c66624c0f, adoption2647/ac9e187d7b86af6eab9abb4b7d9d340dc929a0417ea8e8543d3dde051e2f8133. ROOT FIRST/SECOND 실제 Source 리뷰가 완료되었고 sole R1은 불필요했다.

S15 genuine full4는 FINAL124/160.922초이며 stdout F.., stderr0, JUnit 없음이다. 소유 프로세스 종료·예약 해제·양쪽 EOF·unknown{}가 확인되어 실제 비용을 한 번 기록했다. B/s15-actual.json3593/f19a8f50b799a4a620e2a33128c125b4642d2974e8a7bf7a3980b53a59011bc6 및 B/s15-observations/index.json8262/bc6283d9cf369c7b9f93c95d41b14f99300519e807700945e5101e2629216db6를 보존한다.

Native parity는 시간 초과 없이 return1로 끝났고 lock, typed-state, legacy-control, cache-copy-drift-policy-four-mutations 네 case PASS를 남겼다. 이어 iterative JSON negative에서 Retained reader rejects malformed JSON이라는 assertion이 실패했다. 정확히 어느 입력이 실패했는지는 아직 관측하지 않았다. 두 번째와 세 번째 pytest node는 완료 점을 남겼으며 Native recovery/allocation/atomic은 PASS이다. 네 번째 node는 완료되지 않았다. 실제 Original body는 실행하지 않았다.

고정 정본은 B/s15-bootstrap-settled309-HOLD-20261009.json 및 parent-e358-execution-ledger.json484147/59c6402e1be0124e590b5e4d9f6ebe5622f53ee7df3a63cb97e3401aebdd7543이다. ROOT는 전체309 debits/191 calls와 ordered binary64 totals를 독립 복원했다. sole HOLD는 direct-science-20261009-contact-s15-bootstrap이며 다른 ownership/reservation/pending maps와 jobs/reserved는 비었다. 실패 결과·160.922 비용과 S14 repair400 pointer를 포함한 모든 기존 내용은 보존한다.

## 수정 1: 정확한 negative 입력 구성

tests/test_c_banc_checkpoint_harness_runtime.py의 기존 first node 안에서만 고친다. 현재 negative 배열의 마지막 depth 입력은 ('['*65)+'0'+(']'*65)로 전체 연결식의 괄호가 없다. ROOT는 이 표현의 배열 연산 우선순위가 의도와 다를 가능성을 진단했으며, 실제 실패 입력을 확인한 것은 아니다.

65단계 JSON 문자열을 별도의 변수에 전체 괄호로 명확히 만들어 배열에 단일 원소로 넣는다. 원래 일곱 입력의 문자열·순서·duplicate/escaped-duplicate/nested/trailing/comment/NaN/depth 의도를 보존한다. 개수7와 해당 depth 문자열 길이131·65개 opening/closing을 명시적으로 확인하고, 기존 old/new reader 호출·거부·exact duplicate error·실제 K-Read·clone controls와19 Same matrix를 모두 유지한다. 오류 메시지에는 ordinal과 입력 문자열을 넣어 다음 실패를 분간할 수 있게 한다. assertion 삭제/완화/negative→positive 변경은 허용하지 않는다. 기존 최대 JSON depth와 duplicate 처리도 바꾸지 않는다. 실제 reader가 이 정확한 일곱 negative를 거부하는지는 유료 full4에서만 확인한다.

## 수정 2: immutable proof cache의 반복 JSON 검사 제거

현재 Native K-Recovery는 모든 positive마다 fresh recovery/Source/c5/binding/provenance/policy/grant와 catalog171의 전체 target bytes를 인증한 뒤, K-Clone을 사용해 고정 anchor/policy/grant proof를 매번 serialize→JsonDocument duplicate walk→ConvertFrom-Json로 복제한다. K-RecoveredSnapshot에서는 반환 anchor를 다시 K-Clone한다. JSON duplicate walker를 같은 이미 검증된 고정 문자열에 재적용하는 비용이 남아 있다. 이 비용 기여는 아직 측정하지 않았다.

제안은 오직 K-Recovery의 immutable cache representation을 고정 JSON 문자열로 바꾸는 것이다. cold miss에서는 K-PinnedAnchor의 whole typed 검증을 기존대로 수행하고 전체 proof를 Serialize한 뒤 기존 K-Json으로 한 번 strict parse/duplicate/depth/typed parity를 검증한다. 그 정확한 검증된 문자열만 cache에 보관한다. warm return은 그 문자열을 ConvertFrom-Json -AsHashtable로 새로 parse해 detached typed tree를 만든다. 문자열 자체는 immutable이고 반환된 event는 매 호출 fresh validated event를 결합한다. 반환 객체·anchor/policy/grant의 mutation이 cache에 영향을 주면 안 된다.

기존 K-Clone 일반 함수, K-Json, Same, Conserved-Json/core, K-SourceRaw typed byte[][] 및 기존 controls는 그대로 둔다. cache 전체 문자열 hash를 entry와 함께 저장·검증하는 것이 필요한지 Astra가 평가해야 한다. 해당 비용·공간이 부담이면 보안/계약을 약화해서 진행하지 않는다. cache 내부 위조·타입 변화에 대한 실제 negative control도 같은 first node 안에 추가할 수 있다. cold miss와 warm hit의 전체 typed 결과를 기존 구현과 비교하고 반환 mutation·exception clear·guard ownership/reentry·다음 정상 호출을 유지한다.

이 제안은 fresh Source/c5/event/catalog/policy/grant/provenance를 캐시하거나 raw 읽기를 생략하지 않는다. catalog 전체 identity만으로 target를 신뢰하지 않는다. 캐시 key는 root/schema/epoch/Source/c5/fixed pins/policy/grant/row policy의 모든 기존 값이며 S16 schema가 분리된다. current canonical ledger/tail/owners/once fees/ordered totals/allocation/qualification는 항상 cache 밖에서 전체 typed 검증한다. guard는 첫 metadata read 전부터 own finally까지 유지한다. Python 쪽은 일반 deepcopy representation을 유지하고 S16 scope/pins만 연결한다. 아직 성능 개선을 주장하지 않는다.

## 보존되는 전체 계약

변경 scope는 이전11 replacement/0 additions이다: 두 runtime module, 두 fixture, checkpoint harness tests, journal whitelist, conserved_resource.py count, B/account.ps1, b5/c5/t5. 나머지860 Source, installed93, inherited-large18, 보호421·header regions·wrapper138·core14933·legacy126162·두 운영 문자열·정확한 extraction sentinel과 실제 operational branch를 보존한다. Account 변화는 정확한 Native inline 하나뿐이다.

Native65536/Python65536/account262144/freeze262144/review262144 byte cap는 올리지 않는다. S15 기준 Native64419/account261918로 account 여유가226 bytes뿐이므로 최종 전체 serialized 크기를 실제 측정한다. 필요한 표현 축약은 새/변경 Native cache 부분의 로컬 변수·공백 수준으로 제한하고 의미·원래 guard·assertions·metadata를 삭제하지 않는다. cap 안에서 구현할 수 없으면 Source 채택을 하지 말고 ROOT에 구체적인 한계를 보고한다. 모듈 분리·추가 helper·Add-Type·동적 compiler·project import는 없다.

네 개 원래 pytest node 및 first node의 원래19 matrix/JSON controls, fresh source/cap/provenance·cold/warm·reentry·mutation·archive/missing/drift·unknown once-fee·duplicate commit·atomic lock controls를 모두 유지한다. S15 reader 0/1/boundary/UTF8/cap+1 controls는 그대로 보존하고 기존 S14 stdout1 및 empty stderr 경로도 보존한다. partial pass나 synthetic Original metadata를 full4/actual Original 성공으로 승격하지 않는다.

## 충분한 예산을 먼저 확보하는 conserved400

현 caps generated global3800/local3064/Original14100, combined17900, remaining direct118.62999999999965/shared126.41798439999957/Original1836.9150391004946이다. 현재200 bootstrap+close envelope가 부족해 그대로 retry하지 않는다.

S16는 동일 총한도 안에서 새 unique event direct_science_checkpoint_repair_s16_400를 통해400을 Original→generated로 옮기는 ROOT owned metadata action을 설계한다. caps는4200/3464/13700이 되고 remaining direct518.6299999999997/shared526.4179843999996/Original1436.9150391004946이다. 이 숫자는 arithmetic prospective values이며 실제 action/readback에서 ordered totals와 binary64를 확인한다. 기존 S14 event 재적용·refund·reset·새 grant는 아니다.

원래 settled309 전체 raw/typed를 독립 확인해 새 compressed post309를 미리 CreateNew로 게시한다. 변경은 caps3/remaining2/new pointer1 외 전혀 없어야 한다. action은 bounded common metadata owner token, exact before hash, no foreign owner/uncertainty, pre-cap before every read, durable atomic replace/whole typed readback를 지킨다. 알려진 FINAL과 정확한 own token만 해제하며 uncertain partial/recovery는 실제 durable state를 재확인하고 절대 동일 transfer를 재실행하지 않는다. S14 ROOT admin의 실제 receipt 표현 오류와 사후 settlement를 그대로 보존하며 이번 receipt array arithmetic은 괄호로 묶는다.

실제400 이후의 whole settled309/HOLD를 ROOT accepted immutable S16 original=anchor로 고정한다. 따라서 S16 runtime recovery는 이미 적용된 pointer를 확인하고 repair_delta=[]이며, 추가 transfer를 하지 않는다. 새 event의 immutable prospective publication과 실제 applied receipt를 구분한다. Catalog는 post309를 포함한 고정 closure의 모든 raw target를 검증하며 explicit whole309 trust가 historical semantic replay를 대체하지 않는다는 한계를 유지한다.

S16 expected counts309/191→bootstrap310/192→qualification314/196→body315/197→audit316/198이다. Only genuine sameSource full4 FINAL0/exact JUnit4pass/known closure/once actual fee permits 기존700 transfer. 그 뒤 caps4900/4164/13000이며 원래 bootstrap180/160/20, qualification40/35/5+35/30/5+170/155/15+420/400/20 총690, actual Original300/285/15+120/100/20+close180 총600을 유지한다. 최악bootstrap180 이후700을 합쳐 direct1038.63/shared1046.4179844로690을 충족하고 Original736.9150391004946로600을 충족한다. 모든 actual fee는 genuine dispatcher milliseconds로 기록하며 unknown를0으로 보지 않는다. 실패 full4에서는700·qualification·body를 실행하지 않는다.

## S16 catalog·Source·검증 및 개발 순서

S15 complete catalog171/161/3와 모든 prior immutable metadata/Source archives를 보존한다. 새 catalog는 old171 rows를 먼저 완전히 동일하게 유지하고 추가된 S15 history·actual failure/cost/request/outer/trace/0·3-byte streams·old Source/review/provenance·actual Source11 archive·새 original/post/repair evidence 중 실제 transitive reader closure를 path/cap/kind별로 추가한다. descriptor 자체와 target raw, raw와 JSON kind, zero-length exception과 finite named ledger caps를 구분한다. old0 three는 유지하며 actual S15 empty stderr가 필요하면 네 번째 exact raw0로만 확장한다. current S16 Source/c5/provenance는 catalog 밖 fresh authentication으로 self-cycle를 피한다. ROOT와 Astra가 실제 Source callsites와 old/new rows를 audit한 뒤 finalcatalog를 사용하며 guessed count는 고정하지 않는다.

ROOT draft→actual GPT-6 Astra/max review→조건을 ROOT reflection에 반영→actual GPT-6.1 Sol/xhigh private FIRST→ROOT full review→문제 있을 때 같은 Sol sole R1→ROOT SECOND/필요한 residual→full871/11/0/93/18·보호·whole inline·review pins·caps·metadata auth→별도 actual400 whole readback→Source adoption→fresh recovery와 admission→genuine paid full4 순서다. private flag는 ROOT 종료 전 false이며 root_reviews/sol_reports는 실제 파일 pins만 채운다. 순환 self-hash나 가짜 review는 없다.

설계·Source 리뷰 중 project/Python/import/parser/AST/compile/pytest/ruff/smoke/helper/profile/CUDA/Original probe를 하지 않는다. full4가 끝난 뒤에만 actual evidence를 읽고 실제 cost를 기록한다. actual complete PASS 후700→원래 lint/math23/gateA25/gateB36 qualification→actual Original body/audit→고정 F1–F8과 M2/UI 회귀를 검증한다. 네거티브 오류 입력 수정과 성능 가설이 모두 실제 관측으로 확인되기 전 보행 후보를 채택하지 않는다. 목표는 계속 incomplete다.

Astra는 JSON negative expression의 Source-level 판단, 검증된 immutable string cache의 등가성/보호 한계,226-byte 여유 안의 구현가능성, 새400→700 전체 envelope와 catalog closure를 독립 리뷰한다. 더 작은 안전한 수정 또는 분리된 유료 진단이 필요하다면 구체적인 비용·장부·Source 연결을 제안하되 고정 통과 기준을 낮추지 않는다.

