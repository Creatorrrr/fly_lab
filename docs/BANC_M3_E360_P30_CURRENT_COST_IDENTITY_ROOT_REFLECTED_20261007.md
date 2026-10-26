# P30 실제 Astra 리뷰를 반영한 ROOT 구현 설계

2026-10-07. ROOT는 실제 GPT-6 Astra/max 리뷰 JSON30887B/SHA`ba7f61703cc4a2fbd606fdb4d623d7f1043b32db6dcc59f1589c18d967e1dde8`와 MD4818B/SHA`d4299de1c24fb822dde3d44a11aa12bfbe4999293bb991c6d2e984f0df1f9987`를 읽고 R1–R3 모두를 채택한다. 원 설계 `BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_DESIGN_20261007.md`와 선택43 입력의 보존·범위·자원 조건은 그대로 유지한다. 이 문서의 구체적인 연결 지시를 구현에 적용한다.

목적은 실제 비용 분류 오류를 제거하고 필요한 검사를 통과해 신체 실험으로 이어가는 것이다. Source 선언·테스트 수·문서 출판은 자연스러운 보행이나 생물학적 기전의 증거가 아니다. 실제 P29 Ruff는 성공했으며, Native scanner 준비 실패와 ROOT admin wrapper 오류는 별도 기록으로 유지한다.

## R1: 현재 비용 분류와 실제 첫 유료 행

Native4172/Python3881의 숫자 기본값28을 제거한다. 완료된 역사 범위에 실제 P29 index108..109를 추가하고 원래 logical policy12804/24bd4b5b…와 immutable physical policy를 사용한다. 명시된 역사 범위가 없으면 index>=compiled current start109 **그리고** compiled current pass ID·descriptor가 일치할 때만 현재 행으로 분류한다. 미분류된 과거 index, 잘못된/future ID, type·descriptor 변조는 거부한다. 새로운 default30으로 단순 치환하지 않는다.

현재 start/pass/descriptor는 actual cost gate·Call-Policy·bootstrap/event/fresh pass/freeze에 일관되게 연결한다. Native switch는 역사 descriptor만 선택한다. 매 현재 행마다 전체 역사 증명을 다시 실행하지 않으며 persistent/global ledger cache는 추가하지 않는다.

tests389–394의 역사104 비용을 현재처럼 바꾼 synthetic positive를 실제 P30 Ruff row109 positive로 교체한다. 같은 기존 named test와 PROGRAM16에서 원래 actual cost/actual/request descriptor·charged call을 읽고 Python `validate_cost_authority`와 Native `Assert-Cost-Authority`에 원래 기록을 전달한다. 기대값은 compiled current identity에서 독립적으로 정한다. 실제 첫 Ruff가 없는 경우 prerequisite 오류로 닫으며 skip·합성·private preflight로 대체하지 않는다.

fixture/suite 단계에는 후속 call이나 현재 role hold가 있을 수 있다. 이 작은 검사는 전체 live ledger의 ownership0/next1/end110을 요구하지 않는다. immutable historical109 prefix와 index109의 once charged Ruff·pass30/current descriptor/current own9·원래 request를 좁게 검증한다. 일반 production의 ownership/admission 검사는 별도 기존 경로에 유지한다. 실제 역사P29 row108 positive와 whole109/full-cost/paid105–108/정책/roster END도 보존한다. local wrong-index/pass/descriptor/future/type negatives,84 ordered names/252 phases,37 extraction slots,Native15+same PROGRAM16/one process8초를 유지한다.

## R2: 성공한 전임자와 Source 실패의 계약

새 currentP30 policy/request/pass에 **paid predecessor success**와 **unpaid Source admission failure**를 별도 exact descriptor 필드로 둔다. Native literal/manual/fresh/prospective admission뿐 아니라 실제 pass writer6577–6581, Python5015–5031, prepare54–58, capture/request 및 모든 현재 소비자까지 연결한다. 한쪽에서만 schema를 바꾸지 않는다.

새 전환은 실제 P29 성공한 Ruff0/2.057/cap3/charged_once/internal_valid/collector/capture 유효, exact closed109/P29 next1/failed_cost null/failed_cap0/ownership0, genuine Native scanner 준비 실패·first ErrorRecord·FINAL, 실제 변경된 Source를 검증한다. 역사 Concrete-Failed-Predecessor 규칙은 변경하지 않는다. 성공한 비용을 actual_failure로 이름 붙이거나 과거 P27 실패를 즉시 전임자로 재사용하지 않는다.

역사 P29 pass의 actual_failure=P27/predecessor_failed_cap3은 원래 입장 기록으로 그대로 둔다. P29 cost의 internal_completion=null/internal_completion_valid=true/capture_return descriptor/empty capture errors도 그대로 보존한다. P27의 false/null/malformed capture 규칙을 새 성공 증명에 복제하지 않는다. 역사 P29 frozen Source/policy aliases와 Ruff Call-Policy3/3/885를 실제 비용과 연결하고 과거 모듈은 실행하지 않는다.

실패 문서의 actual_closed109는 당시 logical live-path descriptor다. P30가 live를 변경한 뒤 그 옛 hash를 mutable live에서 읽지 않는다. `parent-e360-p29-after-Ruff-closed-ledger109.json`의 동일한 immutable physical tuple에 명시적으로 연결한다. 생성된 scanner request는 metadata이며 등록이나 지불 이력이 아니다.

## R3: 새 첫 Ruff의 실제 비용 기록 호출

P29 ROOT committed-admin-caller는 Ruff가 이미 지불되어 허용집합에서 뺐다. 새 P30 caller/ABI는 새 첫 Ruff도 허용하고 **필수 exact CaptureReturn descriptor**를 받도록 한다. namespace만 치환하지 않는다. 원래 opaque Native summary를 출판한 뒤 path/typed bytes/SHA descriptor를 account에 전달하고, 순수 Capture-Proof는 내부에서 read된 값으로 검사한다. Native 18-digit identity를 JS 숫자로 parse하지 않는다.

실제 script 예외와 원래 first FINAL은 보존한다. admin branch에 없는 native exit를 LASTEXITCODE로 추측하지 않는다. 실제 반환 budget·once charged cost·원장 상태를 확인한다. 실패 role을 정확히 지불한 것과 다음 role의 허용을 구분한다. post-commit 확인이 실패하면 known committed와 uncertain commit을 구분하고 원래 오류를 보존하며, account/admin/Ruff를 재실행하거나 비용을 바꾸지 않는다. scanner/helper의 원래 completion·collector 계약은 유지한다.

새 caller는 다른 Source 제안과 함께 실제 ROOT 검토·freeze 후에만 genuine 다음 실행에 사용한다. 원래 P29 caller/오류/FINAL과 성공한 비용은 불변이다.

## 실제 H1180와 자원

실제 리뷰가 확인한 old1160+exact20 P29 physical union을 채택한다. ROOT가 별도 immutable 파일로 같은 bytes를 freeze한다. Source 표는216320B/SHA`ce69cf7e48a2c0f7b1dae18deffb784b05f09ae182ae4379059298e53a5e7743`, member1180/aggregate67260839/D512/combined1692다. 정확한20개는 closed109/freeze/frozenpolicy/Source4/Native4/collector/Ruff8이다. old1160 tuples는 변경하지 않는다. table262144B/maxleaf2MiB/fixed18 reader의 독립2MiB aggregate·두 journals·root/clock/poison/fences/unknown-empty/ownership-before-membership을 유지한다. 이 선언은 runtime graph 인증이 아니다.

tool4477–4525의 유한 direct proof 패턴을 사용하고 concrete_fix/review/nested descriptor는 exact typed metadata로 유지한다. 필요하지 않은 재귀 읽기·새 reader/cache/eviction/scope split/table-miss dynamic fallback을 만들지 않는다. P29 stdout19와 stderr0의 정확한 tuples를 기존 completed-stream 경로에 등록한다.

현재 전환이 실제로 읽는 unpaid Source error의 물리 입력은 정확히 세 개다: failure2731/b10df607…, genuine return1606/18b3d770…, first ErrorRecord2386/b3d497c2…. 이를 lookup 전에 current D authority에 선언하고 역사 지불 이력으로 만들지 않는다. nested operation/review·unadmitted request는 metadata만 유지한다.

실제 ROOT baseline79170B/SHA`27a9c96b9a5e36ebdcc7d1764ca37d8a89da69db5577e9464ed62aba32dc28ed`은 protected322/old Native89/current own9/selected43/GOAL/live closed109를 확인했다. old Source와 Source-only residual caller까지 그대로 유지한다. 현재30/future31 denied,pair2/grant0,109/59/26→등록109/60/27→첫debit109/count110이다.

shared676.3040156000001/local565.347,limits1700/1589,잔여1023.6959844/1023.653,whole768+120=888/local margin135.653,new grant0/resetfalse를 유지한다. 원 설계와 리뷰의 모든705/caps/suffix/MAX/future/fixture/suite/host/output 수치를 그대로 보존한다. Original admission false와20초 부족도 유지하며 필수 검사 통과 후 실제 startup/body/audit allowance를 맞춘다.

## 구현 인계와 최종 목표

같은 실제 GPT-6.1 Sol/xhigh에 FIRST를 맡긴다. 변경된 계약이 입장·기록·후속 읽기까지 이어져 난도가 높다. actor는 새 P30 Source4/Native4/collector/policy/caller/boundary/proof/diff/ABI/handoff 제안만 작성한다. canonical/live ledger/GOAL/active plan/history/기존 FIRST/R1과 실패·성공 증거는 수정하지 않는다. 각 현재 semantic site와 실제 인자·first-current-row 전제조건을 인계하고 안정된 FIRST 후 멈춘다.

ROOT FIRST → 필요 시 같은 Sol sole ONE_R1 → ROOT SECOND → 남은 ROOT 수정 → actual apply/freeze/activation/ordered roles 순서다. Source-only 단계에서는 Native text/UTF8/JSON/SHA/regex/ParseFile metadata만 허용하며 대상 Python/AST/import/compile/Ruff/test/PROGRAM/account/helper/function/ScriptBlock/private probe/network/Original arrays·headers·checkpoints/GPU/body/UI를 실행하지 않는다. runtime/8초 적합성/gait는 NOT_RUN이다.

필수 검사와 적합한 Original allowance 이후 같은 후보0→99→300/1.5초 실제 신체 screen과 전체 F1–F8을 계속한다. 외부 CPG/교사/위상/궤적/root 교정을 최종 보행에 쓰지 않고 BANC 해부학 provenance·M2/UI·기존 실패·미사용 초기상태 노출을 유지한다. 현재 F1–F8은 INCOMPLETE다. 사람의 자율 진행 지시에 따라 별도 승인을 묻지 않는다.
