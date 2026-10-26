현재 controlling v3는 Sol에 배정하기 전 v2의 catalogue 숫자 치환이 budget suffix705까지 바꾼 문서 오기를 원래 실제 schedule로 정정한 버전이다. v2는 미배정/미구현 기록으로 보존한다. Catalogue712/combined1224만 바뀌며 roles9/caps/suffix/futureMAX825/705 및 whole888은 원래 실제 값 그대로다. 새로운 설계·예산·runtime을 추가한 것이 아니다.
# ROOT controlling reflected design v3: 실제 누락7 request를 명시적으로 포함

이 파일이 현재 전체 controlling design이다. 최초 설계/실제Astra/ROOT반영v1/705 catalogue/Sol coverageSTOP은 원형 보존한다. 같은 R2 소유권 구조에서 catalogue만712 exact members로 정정했고 Source 구현은 아직 시작하지 않았다.

실제 Sol pre-implementation STOP20389B/SHAa298def26d1eb7feb858b2a1d89bfc87f8b9f42acb039e2d999bd42960862ccf는 closed107.e360_calls57개 중7개 request가705목록에 없음을 확인했다. P2fworker/sworker,P6fworker,P7fworker,P8fworker/sworker-final,P11fworker-final의 정확한 saved descriptor와 실제 physicalUTF8/bytes/SHA가 일치하며 기존 paid_request_identity/validate_live_calls→read_spec→consumed의 역사 의존성이다. 실제 runtime depth/count는 측정하지 않았다.

ROOT가 그 정확한7개를 명시적으로 추가한 refinement4580B/SHA3c377cc1e46660ed6c35eb7169dc2465962ddab4a33d6d69c2b736f6fa67de5a를 출판했다. refined catalogue128270B/SHAdc383ba99574be81306be2f2139829207121b2722f5a730c89b7e721877efeb0, members712,aggregate24263292B,각leaf2MiB,JSON≤262144B다. 이전705metadata126895B와 preparation269700B는 과거 Source 준비 기록이며 새712개 목록을 이미 만들었다는 증거로 쓰지 않는다. 57개 saved requests 모두 새목록과 typed identity가 일치한다. 전체 다른 former-dependency coverage는 여전히 NOT_CERTIFIED다.

Sol은 이 Source audit를 계속하고 완전 coverage가 확인되면 FIRST를 구현한다. 또 missing physical historical dependency가 나오면 arbitrary discovery/동적fallback/목록증액을 하지 말고 exact STOP 보고를 한다. ROOT가 그 근거를 먼저 반영한다. 이것은 Source coverage 단계의 계속이지 ROOT FIRST 뒤 soleONE_R1을 사용한 것이 아니다. 현재 dynamic512+fixed712의 consumed subset,combinedmaximum1224는 기존total512의 명시적 개정이다. 실제 body/F1-F8/M2/UI/원본시간/예산/신경/physics는 변경 또는 검증되지 않았다.

아래 전체 설계의 catalogue값은 현재712값으로 정정한다. Source-preparation proof는705에서 시작했다는 위 경계를 우선한다.
# ROOT가 Astra 리뷰를 반영한 descriptor 소유권 계약 — 2026-10-06

이 문서가 P26 Source 구현의 전체 controlling design이다. 최초 `BANC_M3_E360_DESCRIPTOR_SCOPE_REPAIR_DESIGN_20261006.md`와 실제 리뷰 입력은 원형 보존한다. 전체 M3 및 사용자 지시의 실제 body milestone을 계속한다. 추가 승인 질문은 하지 않는다.

실제 GPT-6 Astra/max 리뷰는 `parent-e360-p26-actual-astra-review.json`18796B/SHAc4913ccc26ae7017e6c7b6e7fd732468ceed4675255f3cb1ce72ca8626b49199, MD5564B/SHA92def405c71b8506307336e0fe0daab6d4c5bc90ce2ff56acab33b019b4c8029다. ROOT가 전체 필수 R1–R3, Source 근거, 한도와 제한을 재독했다. 최초 설계는 REVISE_REQUIRED다. 아래는 그 반론을 반영한 ROOT 결정이며 이전 설계가 승인되었다고 표현하지 않는다.

## R1: 잘못된 원인 해석과 수정안 철회

public `consumed()`의 depth0 early return과 private `_consumed()`를 ROOT가 혼동했다. private `_consumed()`에는 depth 검사 없이 canonical dedupe와512 등록이 있고, positive historical stream도 이를 직접 호출한다. empty/positive 모두 depth0에서 등록한다. `enroll_json_descriptor`와 current exact input에도 의도적인 depth0 등록이 있다. empty만 depth>0일 때 등록한다는 안과 positive가 depth0에서 미등록이라는 전제는 철회한다.

실제 tests920–927은 depth0의 production empty reader→파일 변경→실제 fence rejection/poison을 요구한다. 이 검사를 삭제하거나 반대로 바꾸지 않는다. 즉시 두 번 읽는 안정성만으로 반환 뒤 변경 검사를 대체하지 않는다. bytes0의 exact historical allowlist/SHA-empty/contained strictUTF8/stable two reads/ADMIN_CHECK/ENTRY clock/poison을 모두 유지한다.

## R2: 하나의 scope 안에 한정된 고정 역사 목록과 동적 입력을 분리

현재512 검사 실패는 이미 등록된 같은 경로의 반복 때문이 아니다. Source에서 동일 canonical path를 먼저 dedupe하고 새 경로에서만 한도를 검사한다. 실제 실패 stack은 그 새 empty path 분기에 도달했다. 정확한 runtime depth/전체 count를 별도 측정하지 않았다.

ROOT는 **명시적 계약 개정**을 선택한다. 한 actual scope/동일 ENTRY에 현재 및 일반 literal 동적 manifest512와, 정확하게 열거된 과거 descriptor catalogue를 둔다. 각 catalogue leaf를 실제로 소비할 때 별도의 monotone owned-history 집합에 등록한다. 모든 applicable safety fence는 두 소유 집합의 모든 소비된 leaf를 같은 기존 cold leaf 검사로 확인한다. parsed-clear와 descriptors_only_execution 뒤에도 양쪽 소유 목록을 유지한다. history 값을 cache하거나 성공을 memoize하지 않는다.

이것은 기존 **총512경로 계약의 보존이 아니다**. 동적512 + 고정 역사712에서 총 최대1224개의 서로 다른 소비 경로를 소유할 수 있는 제한적인 개정이다. catalogue 소속과 ordinary/current 소속은 canonical path 기준으로 충돌하지 않아야 한다. catalogue member의 bytes/SHA/path가 충돌하면 poison/fail-closed하고 동적 공간으로 옮겨 허용하지 않는다. 명시적 historical 등록의 unknown member/누락도 실패한다. 일반 미등록 current/literal descriptor는 기존 typed admission과 동적512 한도를 그대로 받는다. overflow를 역사 공간에 자동 spill하지 않는다.

고정 catalogue는 ROOT의 Source 전용 준비 결과 `parent-e360-p26-ROOT-flat-history-refined712.json`이다. 정확한 descriptor는128270B/SHAdc383ba99574be81306be2f2139829207121b2722f5a730c89b7e721877efeb0, member count712, leaf aggregate bytes24263292, 각 leaf 최대2097152, catalogue JSON은 기존 generic262144B 상한 이내다. 회원의 실제 count와 bytes는 고정 목록의 값이며 arbitrary larger history cap을 더하지 않는다. 정책과 compiled metadata pins는 이712개 목록을 정확히 묶는다. 실제로 소유되는 subset은 monotone이며712를 넘거나 새 회원을 발견해 확장하지 않는다.

준비 근거 `parent-e360-p26-ROOT-flat-history-Source-preparation.json`269700B/SHAe18d7f9fb03a08b6d5a9ac14150789d2936d6684fc737326f4683640f3847ae0는 Source 감사용이다. 이 감사 문서는 production generic JSON reader에 주입하지 않는다. ROOT는 named Source 두 개의 실제 JSON descriptor literal leaf, baseline protected322/oldNative64, actual P25 frozen Source4/Native4/collector 및 named artifacts, closed107 debit와 지정된 cost-field descriptors만 준비 단계에 사용했다. runtime recursive JSON/filesystem 탐색은 추가하지 않는다.

current mutable own9/활성 policy/live ledger/세 활성 문서는 목록에서 제외했다(36 seed occurrences). 명시적 absolute library Source93개는 기존 full11/child14 foreign-source 검증 경로를 유지하며 root-relative FixedAuthorityScope 역사 회원으로 바꾸지 않는다. E318 original raw audit4033033B 한 건은 원래 해당 scope의2MiB leaf 상한 밖이라 제외했다. 원본 binary/array를 열거나 hash하지 않았다. Source proof에서 이 제외 분류와 실제 formerly consumed history dependencies의 **완전 coverage**를 확인해야 한다. 준비 목록은 coverage/timing PASS가 아니다.

Sol은 Source FIRST 구현 전에 이 finite catalogue가 기존 실제 historical dependency 경로를 모두 소유할 수 있는지 정적 Source/명시적 메타데이터 근거로 확인한다. 필요한 physical historical member가 빠졌다면 임의 discovery/동적 fallback/카탈로그 증액을 하지 말고, exact missing descriptor와 근거를 보고 STOP한다. ROOT가 다시 목록/설계를 명시적으로 반영해야 한다. 현재712 목록의 완전 coverage가 확인된 후에만 아래 Source 수정안을 완성한다.

catalogue는 Source의 metadata declaration 또는 검증된 current policy의 정확한 flat descriptor 데이터로 연결한다. metadata pins 보유는 허용하지만 parsed historical documents, budget/current selector, admission/fence 성공 cache는 금지한다. 현재 Source hash를 역사 회원에 넣지 않아 hash cycle을 만들지 않는다. 파일 table을 실제로 읽는 설계라면 그 table의 exact descriptor 자체도 동적 manifest에 등록해 모든 후속 fence에서 변경을 검사한다. Scratch-root 테스트가 다른 root의 실제 파일이나 이미 검증된 성공 상태를 빌리지 않도록 한다.

기존 공통 descriptor 등록과 실제 fence 루프를 최소 수정한다. `_consumed`, direct JSON enrollment/current input, positive/empty historical stream 등 동일 실제 등록 경로에 연결한다. canonical/typed identity와 relative-contained path 규칙을 유지한다. reader의 literal descriptor lookup은 양쪽 owned 집합을 정확히 조회하고 중복/충돌을 허용하지 않는다. 실패 시 두 manifest를 제거하거나 poison 상태를 회복하지 않는다. 실제 operation 종료의 cleanup에서만 소유권을 해제한다. 새 fixed reader/범용 validator/historical module clone/subprocess/harness/preflight를 추가하지 않는다.

모든 leaf에 기존 path/stat/bytes/SHA/strictUTF8/two reads/clock을 같은 ENTRY 안에서 수행한다. prepublication/post-durable close/readback, parsed-clear 이후, pre-Popen, pre-array 및 failure publication의 원래 fence가 두 집합에 그대로 적용되어야 한다. 즉시 production typed semantic history proof도 uncached로 유지한다.

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
