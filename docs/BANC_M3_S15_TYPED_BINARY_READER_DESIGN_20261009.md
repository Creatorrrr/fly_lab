# S15 ROOT 설계 — 2026-10-09

목적은 BANC M3 실제 신체 보행 검증으로 진행하기 위해 검증 병목을 줄이는 것이다. F1–F8, 신경 출력/물리 결합, 외부 CPG·phase·teacher·root 보정 금지, startup99/8개 입력/short 기준과 M2/UI는 그대로다. latest human의 자동 진행 권한에 따라 별도 승인 질문 없이 기존 총예산 안에서 진행한다.

## 실제 실패와 다음 가설

S14 실제 full4는 INCOMPLETE/exit124/160.841초/stdout F/stderr0/JUnit 없음이다. s14-actual.json3542/c42eaa6d304a32ecc2900d52e3d49f496c79249b4f90cf457448a79180d9d49e, actual Source s14-s1.json262138/af01b56a98ac6611aafadcb40f440fe7ba31631505871901befd82d7ad27eddb를 보존한다. 실제 Native raw observations는 s14-observations/index.json3686/962426af16db7b83f4834f567438951d275cb8ad8bd2473a48dccdf706ba754c이다. synthetic Original metadata만 PASS했고 실제 Original은 실행되지 않았다. Native 첫 batch의 lock/typed-state/legacy-control 세 case receipt 뒤 cache-copy-drift-policy case가 같은 parent ENTRY deadline에 끝나지 않았다. primitive19와 나머지 세 node/full qualification은 미검증이다. 첫 F의 전체 traceback은 끝나지 않은 pytest가 출력하지 않았다. 별도 빠른 실행/부분 test/새 profiler로 원인을 확인한 사실은 없다.

고정 기록 재구성 반복을 줄였지만 전체 통과나 시간 개선이 증명되지 않았다. 현재 K-SourceRaw의 매 두 번 읽기에서 $j=@(); $j+=,@($m.ToArray())를 사용하고, 버퍼를 New-Object byte[] 65536으로 만든다. byte[]를 PowerShell array expression/concatenation 경로로 전달하는 표현 비용을 다음 수정 가설로 삼는다. Microsoft 현행 about_Arrays는 무타입 배열이 Object[]이며 강한 타입 배열을 지정할 수 있다고 설명한다(https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_arrays?view=powershell-7.6). 실제 기여도와 예상 배속은 측정하지 않았다.

## 좁은 최적화

K-SourceRaw의 로컬 두 결과를 [byte[][]]::new(2)로 고정하고 각 결과를 $j[$i]=$m.ToArray()로 직접 저장한다. scratch buffer는 [byte[]]::new(65536)로 직접 만든다. 바이트 배열을 @()/+=로 펼쳤다가 다시 cast하는 경로를 없앤다. 반환은 단일 byte[] 그대로이며 빈 byte[]도 유지한다. hash/whole repeated raw equality/UTF8 validity/정확한 길이/selected cap+1 bounded streaming/FileShare.Read/파일 시간 및 모든 ancestor reparse 검사/namespace 검사/2회 실제 읽기는 그대로다. 공유 reader의 의미나 public signature를 바꾸지 않는다. 새 compiled helper/Add-Type/새 flat Source key/별도 performance test/reader 호출 건너뛰기/cache로 원본 내용 긍정 판정 금지. 다른 비교/캐시/API 변경은 이 설계에 넣지 않는다. 목적은 같은 full4와 full qualification의 실제 성공이며 Source 변경 자체가 성능 PASS가 아니다.

현재 typed Source/guard controls는 계속 유지한다. whole invocation guard는 최초 read 전부터 자신의 finally까지이고 inner rejection이 외부 guard를 해제하지 않는다. cache는 immutable typed checkpoint만 저장한다. 모든 positive use는 전체 고정 catalog와 모든 raw operands를 재인증한다. current full ledger/tail/ordered binary64 totals/owners/cost/SourceFence/allocation/qualification은 cache 밖이다.

## 실제 whole308를 새 기준으로 삼음

s14-bootstrap-settled308-HOLD-20261009.json483020/c15a4a7237942a612a2c5eb076fcd47be26a6325f02b2264542c76bcbf237d7f를 ROOT가 독립 full typed 재구성했다. original307→exact 실제 conserved400(caps3/remaining2/pointer1)→exact recovery(pointer+알려진 실패 HOLD 해제)→실제 request/known closure/once-fee160.841→whole308/190이다. 불성공을 성공으로 바꾸지 않았고 현재 sole HOLD는 direct-science-20261009-contact-s14-bootstrap의 L stage unsuccessful; successor blocked이다. 다른 owner/reservation/pending maps는 empty이고 jobs/reserve0/공통 lock 없음이다. 모든 이전 비용·실패·Source·catalog·grant·policy·400/700 이벤트 상태를 보존한다.

S15 pinned_anchor는 이 정확한 actual whole308를 명시적 신뢰 입력으로 받는다. original와 anchor 두 이름은 같은 immutable actual308 pin을 가리킬 수 있으며 prospective caps 변경을 다시 구성하지 않는다. 기존 original307/post307/repair400는 역사 lineage로 보존한다. S14_PINS를 S15_PINS로 바꾸되 대응 pin 구조의 original/anchor/catalog/repair/before 다섯 keys와 정확한 before S14 Source/기존 repair400를 명시한다. 새로운 trust literal ROOT_WHOLE308_EXPLICIT_TRUST_NOT_HISTORY_REPLAY. pinned_anchor가 whole308/190, policy/grant, caps3800/3064/14100, ordered totals, known sole S14 failed HOLD, 빈 기타 owners/jobs/reserve0를 검증한다. K-Anchor/HISTORY_CAPS 및 old history functions가 필요한 기존306/3400/2664/14500 문맥은 그대로 유지하고 과거 검증 graph를 새 epoch로 재해석하지 않는다.

S15 recovery는 별도 fresh immutable descriptor이며 exact actual308 before/최종 S15 Source/actual failure/catalog/기존 repair를 바인딩한다. 해당 알려진 S14 failed HOLD만 해제하고 새로운 recovery pointer를 추가한다. full raw before/whole typed after/own metadata token 검사, no new grant/refund/reset. 이번에는 추가 conserved400/admin exception이 없다.

## 고정 raw 목록과 Source

s15-root-catalog-v1.json47499/a22adf8c890daca5a8cacbc05fff60f16b4cd5e784312e0d2a9fedb8fe74a2b2: 기존130 모든 reader rows를 보존하고 latest S14 whole308 lineage17 + 실제 executed Source11 archives를 추가한158 rows/154 paths이다. ROOT는 전체158 target raw/pin/cap/ordinal/conflict/reader tuple를 Source-only 인증했다. 이는 의미 검증 재실행 또는 count만으로 완전성 증명이 아니다. Source-callsite coverage와 actual independent whole308 lineage를 함께 리뷰한다. catalog complete_status는 immutable publication 기록이고 실제 리뷰/contract에 수용을 기록한다.

s15-before-index.json4714/c4b4577a3cd705030088b6cfbd92a3d4732776e74dd3a7d7730be1f84612a756의11개는 Source s14-s1의 실제 실행 tuple과 exact raw MATCH이다. 일반 metadata262144, Python/Native modules 및 해당 literal archive65536, finite named ledger524288, lock16384 그대로다. 정확한 빈 stderr는 L11/L13R2/S14 세 literal path+0 bytes+empty SHA만 raw0 허용한다. 임의 loader/archive discovery/current Source fallback/cap 증가 금지. catalog schema flylab.m3.s15.ROOT-static-historical-catalog.v1; whole trusted catalog pin과 exact keys/ordinal/int-not-bool-float/kind/cap/alias/missing/extra/reorder/conflict 거부를 유지한다. 각 fixture root에 whole158과 모든 provenance/contract operands를 독립 stage한다. 현재 four-node 내부 existing positive/negative controls 및 primitive19/JSON controls 모두 유지한다.

현재 Source 수871/93 installed/18 inherited exceptions,11 replacements/0 additions,protected421/wrapper138/core14933/전체 Native inline1를 유지한다. core와 legacy를 수정하지 않는다. runtime Python/Native ≤65536, account/freeze/review ≤262144, exact serialized caps를 최종 측정한다. 현재 여유 Python1090/Native1643/account752/freeze6 bytes이므로 fit을 추정하지 않는다. Source freeze의 기존 allowed top fields만 변경하고 reviews에 실제 Astra/ROOT FIRST/SECOND/Sol FIRST/sole R1 pin을 acyclic하게 기록한다. ROOT 완료 flag는 실제 Source review 이후에만 true이며 science PASS로 사용하지 않는다.

b5의 과학 구성은 그대로이고 routing/새 epoch/actual whole308 및 latest S14 failure bindings만 갱신한다. t5는 epoch만 갱신한다. c5는 before S14 Source, new actual whole308/canonical lineage/Source coverage/catalog/provenance3를 고정하고 counts와 routing을 명시한다. old J-body와 scientific invariants, criteria, local limits6300/6000, candidate4/alpha.0001/theta/selected/eightinputs/preparation/startup는 바꾸지 않는다.

## 예산과 단계

현재 caps3800/3064/14100, combined17900; exact remaining direct279.5519999999997/shared287.3399843999996/Original1836.9150391004946이다. 신규 grant/refund/400/초기화/spent replay는 없다. 새 same-full4 bootstrap180/160/20 + close headroom20=200이 현재 두 Generated 잔액 안에 들어간다. actual genuine FINAL/known closure/once fee 이후 같은 Source full4 4pass/0fail/0error/0skip+exact JUnit일 때만 기존700 보존 재배분으로 caps4500/3764/13400에 진행한다. 그 후 최악 bootstrap180이어도 direct799.552/shared807.3399844≥full qualification690, Original1136.9150391≥body300+conditional audit120+close180=600이다. close는 reserve headroom이지 가공 fee가 아니다.

fresh calls direct-science-20261009-contact-s15-{bootstrap,lint,math,gate-a,gate-b}, contact-learning-20261009-s15-{body,body-audit}. 기존 cap triples180/160/20,40/35/5,35/30/5,170/155/15,420/400/20,300/285/15,120/100/20 유지. baseline308/190→bootstrap309/191→qualification313/195→body314/196→audit315/197. full math23/A25/B36/indivisible groups/full lint 뒤 actual Original. 관련 verifier returned debit313와 모든 active offsets를 맞춘다. historical counts는 변경하지 않는다. 추가 테스트 node, shortcut qualification 또는 body 선실행 금지.

## 실행 workflow

실제 Astra GPT-6/max 설계 리뷰→ROOT 반영→GPT-6.1 Sol/xhigh private FIRST→ROOT FIRST→필요하면 동일 Sol sole R1→ROOT SECOND/잔여 수정→full Source/caps/protection/provenance auth/adoption→fresh known308 recovery→fresh admitted genuine full4→PASS에만700→full qualification→Original body/audit 및 F1–F8. 새 설계 단계로서 S14 완료된 R1을 반복 요청하지 않는다.

No free project execution/import/Python/parser/AST/compile/pytest/lint/helper/smoke/profile/CUDA/process/Original probes. bounded PowerShell/.NET Source/metadata 읽기/작성/hash/ledger administration만 허용한다. 실제 프로젝트 실행은 finite fresh admitted genuine collector만 사용한다. 실패는 actual evidence/cost/HOLD를 기록하고 예산/완전한 envelope를 다시 판단한다. 단순 Source/metadata 기록을 M3 진척/goal complete로 간주하지 않는다. Original assets 읽기/stat/hash는 admitted actual Original 전까지 하지 않는다.
