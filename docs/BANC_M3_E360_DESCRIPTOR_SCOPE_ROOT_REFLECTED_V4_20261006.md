# P26 descriptor ownership — ROOT 전체 반영 v4, 2026-10-06

이 파일이 현재 전체 controlling design이다. 기존v1/v2/v3, 최초 잘못된 depth-only 설계,705/712/1123 후보, 실제 Sol coverage STOP와 모든 실패를 보존한다. 구현은 아직 시작하지 않았고 sole ONE_R1은 미사용이다. 사용자 최신 지시에 따라 추가 승인 질문 없이 합의된 개발·리뷰 순서로 진행한다.

ROOT는 최초 실제 Astra R1–R3(c4913ccc26ae7017e6c7b6e7fd732468ceed4675255f3cb1ce72ca8626b49199)를 반영한 v3에 이어, 실제 superset Astra/max 리뷰30270B/SHA3fc2a58d7410616e151011e19375cd19b1ef07cc9ca02988c1eb6ab601406903와 MD6098B/SHA5f05a78bb40904b990012fe3b319f65e17aa5c3546efe3012c082961f5cfd6fa의 필수R1/R2/R3 전체를 읽었다.1123 원안은 REVISE이며, ROOT가 별도 준비한1133 후보의 exact finite membership correction만 실제 리뷰에서 확인됐다. ROOT는 그1133을 선택하고 소유권 분류와 독립 journal 연결 요구를 아래 전체 설계에 반영한다. Source나 실행이 이미 PASS했다고 표현하지 않는다.

실제 Source 전임자/paid 전임자는 P25/closed107이다. Budget suffix705와 모든 caps/whole888는 그대로 유지하며 catalogue count1133으로 치환하지 않는다. 현재 table1133/aggregate62517772/combined1645만 descriptor 계약 개정이다. Source FIRST→ROOT FIRST→필요시same Sol sole ONE_R1→ROOT SECOND→ROOT 잔여→actual apply/freeze/activation/genuine gate를 진행한다. 유효한 필요한 gate 후 same-instance0→99→300/actual1.5s body milestone을 이어가며 full F1–F8는 ACTIVE/INCOMPLETE다.

## R1: 잘못된 원인 해석과 수정안 철회

public `consumed()`의 depth0 early return과 private `_consumed()`를 ROOT가 혼동했다. private `_consumed()`에는 depth 검사 없이 canonical dedupe와512 등록이 있고, positive historical stream도 이를 직접 호출한다. empty/positive 모두 depth0에서 등록한다. `enroll_json_descriptor`와 current exact input에도 의도적인 depth0 등록이 있다. empty만 depth>0일 때 등록한다는 안과 positive가 depth0에서 미등록이라는 전제는 철회한다.

실제 tests920–927은 depth0의 production empty reader→파일 변경→실제 fence rejection/poison을 요구한다. 이 검사를 삭제하거나 반대로 바꾸지 않는다. 즉시 두 번 읽는 안정성만으로 반환 뒤 변경 검사를 대체하지 않는다. bytes0의 exact historical allowlist/SHA-empty/contained strictUTF8/stable two reads/ADMIN_CHECK/ENTRY clock/poison을 모두 유지한다.

## R2: 실제1133 고정 역사 회원과 명시적 소유권

하나의 실제 scope/ENTRY에 dynamic512와 exact history1133의 두 disjoint monotone consumed 목록을 둔다. 총 maximum1645다. 이것은 옛 total512 및 v3 combined1224의 명시적 계약 개정이다. catalogue/source metadata/count/aggregate/pin을 정확히 묶고 arbitrary cap/spill/eviction/reset/newscope/newreader/recursive runtime discovery를 도입하지 않는다.

최종 선택은 parent-e360-p26-ROOT-flat-history-refined1133.json206943B/SHAbee32d4dac74f2daa3d91ae4908df0accd82085d38a357f36ee9b0c5c04f95c6다. Members1133/aggregate62517772B/maxleaf2097152B/tableJSON≤262144B다. 기존712+Sol이 명시한411+actual Source snapshot8+actual policy.closed_predecessor_ledger2의 exact union을 실제 Astra가 독립 확인했다. 원래341/70와 별도 P17 정정343/68의 Source relevance 경계를 보존한다. 불명확한68의 필수성·불사용·runtime 소비를 인증하지 않고 보수적 회원으로 포함한다.

실제 Astra가18 fixed reader의 top-level/명시적 collection·relocation,8 historical contexts의15 physical freeze/policy와 직접 읽는 selected fields,194 concrete-fix occurrences/90 exact pass-context relocation을 대조했고, named 범위에서는 확정 추가10 외 필수 누락을 확인하지 못했다. 이는 Source 구현을 진행할 유한 회원 보정이지 완전한 transitive graph/runtime census/timing PASS가 아니다. Unknown historical input은 구현과 실제 후속 gate에서도 fail-closed한다. 기존705/712/1123 후보와 누락·잘못된 분류·모든 리뷰를 원형 보존한다.

**회원 조회 이전에 소유권을 정한다.** 실제 fixed18 reader, completed-history proof와 기존 exact pass/full-descriptor relocation context가 역사 physical input임을 기존 enrollment 지점에 명시적으로 전달해야 한다. table miss를 current/generic으로 해석하거나 broad path prefix만으로 권한을 정하지 않는다. Historical logical descriptor는 기존 pass/context resolver로 physical leaf를 선택한 뒤 등록하며 Source audit의 global alias map은 runtime resolver가 아니다. Missing member, typed identity conflict, 잘못된 history context는 dynamic 빈 공간이 있더라도 삽입 전에 poison/reject한다.

Current own9/policy/live ledger/활성 문서, table pin 자체, 일반 generic literal과 명시적으로 admitted synthetic scratch input은 기존 진짜 current/generic admission에 따라 dynamic512를 사용한다. Synthetic 등록을 시험할 때는 회원 부재에 따른 암묵적 fallback이 아니라 명시적 generic/scratch admission을 사용한다. 기존 public consumed depth 정책과 private/direct depth0 등록을 모두 유지한다. Historical context annotation은 새 preflight/reader/frame으로 semantic 검사를 우회하지 않고 metadata-only leaf를 강제로 소비하지 않는다.

고정 metadata는 Source declaration 또는 정확히 검증된 flat table로 연결한다. 파일 table을 실제로 읽는 경우 그 자체 exact pin을 dynamic manifest에 등록하고 모든 후속 applicable fence에서 변화를 검사한다. Scratch-root는 자기 root에 admitted된 table/leaf 정체성과 실제 파일을 사용한다. 다른 root의 파일·table validation 성공·semantic 성공을 빌리지 않는다. 모든 실제 consumed history leaf는1133 이하의 별도 목록에 monotone 등록하고, current/history canonical identity가 겹치거나 충돌하면 poison한다. Parsed historical documents/budget/current selector/admission/fence 성공 cache는 금지한다.

Parsed-clear/descriptors_only_execution/poison 중 두 consumed 목록이 모두 살아 있다. 원래 scope fence의 path/stat/bytes/SHA/strictUTF8/동일 두 읽기/clock을 두 목록의 모든 실제 소비 leaf에 적용한다. Fixed-reader literal lookup, duplicate identity 검사, empty depth0의 exact allowlist/등록, 정상·실패 actual cleanup, 모든 publication/pre-Popen/pre-array 경로를 빠짐없이 연결한다. 전체1133 회원을 매 fence마다 강제 읽지는 않는다.

**별도 journal fence도 두 목록을 소유한다.** 실제 canonical Source7637의 boundary.descriptors-only snapshot을 두 disjoint consumed 목록의 합집합 snapshot으로 바꾼다.7649–7661의 independently captured resolve/hash/read/clock cold fence와 same ENTRY/첫 오류/poison을 그대로 유지한다. SUT가 monkeypatch할 mutable 성공 callback으로 바꾸지 않는다. 독립 journal의 실제 late-history drift 검출을 기존 named case에 연결한다.

Current mutable36 seed occurrences/14 names, foreign93의 full11/child14 및 oversize E3184033033B의 분리 경로를 그대로 두며 이번 Source audit로 인증하지 않는다. Raw audit2281597B/relevance713154B/ROOT preparation586217B 같은 Source 증거를 production generic JSON에 주입하지 않는다. 원본 array/binary/header/checkpoint를 열지 않았다.
## 기존 검사와 Source 리뷰

ordered84 이름/252phases/fixed18/ENTRY-depth0-poison/원래 Native15 및 same PROGRAM16/one subprocess8/full11/childSource14/latefourth/nonownQ/current512KiB/generic256KiB/CASE-Q64KiB/receipt34-old33을 유지한다. descriptor 계약만 위에서 명시한 대로 바꾼다. 기존 nested fixed18 literal aggregate2MiB/literal256KiB/deepcopy/root isolation도 유지한다.

기존 named tests 안에서 depth0/depth>0 empty/positive late drift, direct512/513 overflow/poison, duplicate/conflict/unknown member, 양쪽 ownership, parsed-clear survival, prepublication/pre-Popen/pre-array를 같은 production registration/fence로 검사한다. scratch table/파일 입력은 실제 등록 함수를 사용하며 거울 predicate나 이미 성공한 closure stub을 쓰지 않는다. 하나의 실제 complete current/history selector positive가 END에 도달해야 한다. 각 negative마다 whole history를 다시 replay하지 않는다. runtime timing 및 증상 해소는 실제 generated execution 전에는 NOT_RUN이다.

actual hot ledger four-return/_totals/_local/single call와 실제 Native activation predicate/Boolean checked+allowed consumer를 보존한다. Source review에서는 Native text/JSON/SHA/regex/ParseFile만 허용한다. Python/import/AST/compile/Ruff/tests/assembled PROGRAM/private probes/preflight/원본 array/header/hash/checkpoint/CUDA/신체/UI/서비스는 수행하지 않는다.

## R3: 실제 P25 전임자와 현재 P26

P25 실제 apply/activation exit0 후 Ruff1.695s/exit0/cap3와 scanner4.716s/exit1/cap30을 각각 once charged했다. administrative collector/internal validity 둘 다 true지만 expected scanner output/internal completion은 null이며 scanner/Q 성공은 아니다. 실제 closed107199728B/SHA097201fd2c51b8445d0415beea88248713233f127e581c5e3c7233ec9db66f43,107debits/57calls/24paidpasses, P25 next2/failed_cap30, allcharged/ownership0/fallbackunused released다.

Source/paid 전임자는 실제 frozen P25다. 실제 P25 Source4/frozen4/own9/Native4/collector/policy/freeze/reviews/executed operations/activation snapshot/Ruff8/scanner4/closed107/decision을 별도 정확히 보존한다. unpaid P24도 그대로 유지하고 paid range나 snapshot을 만들지 않는다. oldNative69=prior64+actualP25 Native4/collector5와 protected322, ROOT GOAL151552/SHA16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace, human1098/SHA8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224를 유지한다.

P25 original logical current-slot policy9672/SHA77d1fe0a4ac8b9486f5530a8cefba125eed5fc60f0e40c087ddb753ef8d7b6b5와 physical frozen policy를 분리한다. actual104=P23,105=P25Ruff,106=P25scanner,107이후=currentP26이다. synthetic107/current는 separately labeled이고 future 거부값은27이다. 현재 namespace26과 completed24paid passes를 혼동하지 않는다. activation은107/57/24에 pair2/grant0만 추가한다. fresh Ruff 등록으로25passes/58calls, 첫 actual charge로108costs다. P25 자체의 preserved predecessor cap3/P23Ruff는 유지하고 현재 P26은 actualP25 scanner30/next2를 전임 실패로 사용한다.

실제 Native activation 함수 count107와 same PROGRAM16 whole107 positive는 real closed107/current authority/freeze/locktrue/Booleantrue 및 first comparator END에 연결한다. count106/lock-held/current-pair negatives는 실제 함수의 normal Booleanfalse/reached/null failure를 바깥 consumer에서도 요구한다. unrelated setup exception을 reject 성공으로 세지 않는다. 기존 definitions에 실제 함수를 포함하고 새 fragment/mirror/process를 만들지 않는다. 현재 policy/bootstrap/helper/worker/receipt/Q/Native4/collector/pure guard/FULL canonical branch의 exact ABI가 모두 일치해야 한다.

## 자원과 실제 body 전환

현재 shared672.5270156/local561.57 및1700/1589를 그대로 사용한다. 최초 설계의 과도한 shared 소수 표기는 이 reflected copy에서만 정정한다. local 잔여1027.43/whole888 여유139.43이며 새 grant0/resetfalse다. roles9/caps3,30,30,60,20,40,520,25,40; primary768/admin155/fallback120/whole888; primary suffix765,735,705,645,625,585,65,40,0; fallback885,855,825,765,745,705,185,160,120; future125/65/MAX825/705; fixture60/55/51/42/suite520/480/450/360/host2GiB/output16MiB와 firstFINAL/continuous same session/opaque output/NativeInt64ticks/ownedclose/once debit를 유지한다. timing fit은 아직 증명되지 않았다. first closed failure 뒤에는 같은 실행을 재시도하지 않는다.

본 Source 복구는 원본 시간 배정이나 신경/물리를 수행하지 않는다. 현재 original cumulative10020.115960899504/10200, stage126.361/300, local0/20이다. 실제 generated gates가 유효하고 새 구체 문제가 없으면 반복 검증을 확대하지 않고 startup 보존 후보 하나의 동일 인스턴스0→99→300/실제1.5s body milestone으로 전환한다. body 전에 충분한 원본 시간과 저장/close/audit를 실제로 등록하고 사용 비용을 보존한다. 목표 forward≥1.6790085779712631mm/slip≤1.8026436968249955mm/s/bout≥0.3s/각 여섯 다리 qualified complete cycles>0/nativehealth/startupfour를 그대로 검사한다. 같은 θ/checkpoint/decoder/body를 쓰고 reset/teacher warm start/external phase/trajectory/root 교정은 허용하지 않는다.

전체 고정 F1–F8/10s/two unused states/six targets-stop-restart/M2/UI/anatomy/provenance는 유지한다. E358/E359는 technical-invalid/unqualified로 보존하고 적격 neutral로 소급하지 않는다. source/lint/Q/단기 결과를 자연 보행 또는 생물학적 기전 성공으로 바꾸지 않는다. 전체 goal은 ACTIVE/INCOMPLETE다.

실제 Astra 리뷰→이 ROOT 반영→실제 GPT-6.1 Sol/xhigh Source FIRST(coverage先)→ROOT FIRST→필요시same Sol sole ONE_R1→ROOT SECOND→ROOT 잔여 수정→actual apply/freeze/activation/genuine roles를 진행한다. 기존 실제 review/동작 기록을 허구 completion으로 복제하지 않는다.

실제 superset R2/R3 후속 검사: 기존 named fixed-scope/publication/child-journal cases에서 unused dynamic capacity가 있는 unknown-history를 실제 public/private/empty/read_spec 경로로 거부하고 poison해야 한다. 단순 overflow negative로 이를 대신하지 않는다. Explicit generic/scratch admission의512/513, 양쪽 owned identity/parsed-clear/prepublication/pre-Popen/pre-array/journal late-history drift를 실제 production 함수로 확인한다.84names/252phases/samePROGRAM16/oneprocess8s를 유지하며 새 subprocess/per-negative whole-history replay/mirror predicate/probe는 금지한다.
