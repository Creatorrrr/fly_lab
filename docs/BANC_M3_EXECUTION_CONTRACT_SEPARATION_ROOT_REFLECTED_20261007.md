# ROOT 반영: 고정 실행 계약과 현재 시도 데이터

기존 [설계](BANC_M3_EXECUTION_CONTRACT_SEPARATION_ROOT_DESIGN_20261007.md)7986B/SHA66f7bf64d8ab4cf7e83d6aab70031241780b8850427e3435ce4698aded0ec882는 보존한다. 실제 GPT-6 Astra/max의 review JSON24247B/SHA674d65999676d3b29680a7274911d5bc0b18d12d3db32d7c1c6978ac76be78fd와 MD7439B/SHAd750682d6727b98f71c0690812b92870cdd6f09364467114f07c064142aa8925를 ROOT가 읽고 R1–R3를 반영했다. 리뷰는 SOURCE_ONLY/NOT_RUN이고 실행 PASS가 아니다.

## 반영 결론

외부 dispatcher만 추가하는 설계는 불충분하다. Native account의 실제 entry5707/Validate-Call-State6643과 Python main/effective_policy/current_scope/ledger_state/build/closed_exit/Q가 같은 과거 정책을 다시 호출한다. 이 **prospective selection 경계**를 하나의 고정된 current context ABI로 분리한다. 숫자별 새 P31 구현이나 새로운 proof forest를 만들지 않는다. 과거 reader는 원래 명시적 history 요청에 그대로 남긴다.

R1은 ROOT가 기존 P30 residual로 처리한다. 별도의 미적용 patch1448B/SHA8aee8dfb59dd3424a587b0c9616ffb1dc89f57ba123b6ae15ab53e218955c68b가 두 실제 소비자를 수정한다. Sol은 canonical을 수정하지 않고 자신의 제안 Source에 이 patch를 포함해 통합 연결을 설계한다. bytes0는 exact completed-stream reader에서 allow_empty=True/historical journal/두 번 읽기/UTF8/SHA/root/clock/poison/fences를 처리하고 positive-only enrollment를 중복하지 않는다. lib.read_spec 및 dynamic 입력의 bytes>0 규칙은 그대로다. 기존 current policy/closure·owned-scope 테스트에서 실제 두 로그의 성공과 unknown/wrong SHA/type/owner/root/late drift 거부를 검사하며 roster를 늘리지 않는다.

## 최소 current context ABI

고정 schema/rule/source implementation은 manifest의 임의 명령·경로·권한을 허용하지 않는다. finite generated nine-role order, 실제 알려진 실행 argv template, 경로 containment/no-link drift, 텍스트/원본 경계, once cost/ownership/typed status/최초 FINAL 시계를 구현한다. 현재 attempt/call IDs만 데이터다. rule hash→source freeze→manifest→Native admission record로 단방향 연결하고 자기 hash cycle을 만들지 않는다. 아직 context manifest가 없으면 legacy historical 코드로 명시적으로 돌아가며, 잘못된 manifest가 있으면 fail closed한다. 불완전 context를 조용히 과거 경로로 대신하지 않는다.

첫 등록은 actual whole live==immutable closed111221187B/SHAca3c71875072a3a1e94ca15383b902c3911c32e108deea6de8dab86ae19078dc와 all charged/ownership0를 요구한다. 등록 뒤에는 baseline의 old111 debits 순서 및 old61 calls/27 passes/모든 authority/grants/Original/science 값을 보존하고, whitelist로 정한 새 context/call/pass/reservation/pending/hold/cost delta만 검사한다. 매번 whole live==old111을 요구하지 않는다. source freeze와 필요한 physical history/aliases는 봉인된 데이터로 검증한다. historical data의 보존은 과거 실패 검사의 PASS 인증이 아니다.

현재 suffix만 actual request/trace/exit/receipt links, 고유 call/역할 순서, 실패 시 다음 역할 차단, known/pending/unknown ownership, 필요한 동일 pending seal/fallback, 실제 cumulative로 검증한다. raw numerical debit values와 offset42 및 기존 rounding 규약을 유지하며 과거 정책 함수를 재실행하지 않는다. 매 입장 live/anchor/manifest/source/declared physical dependency를 새로 읽고 cold-fence한다. process-wide admission cache, 테이블 miss→dynamic fallback, descriptor 재귀 탐색은 금지한다. H1180 기존 tuples는 그대로 보존하며 실제로 필요한 새 P30 direct physical leaf만 유한 데이터 table amendment로 추가할 수 있다. count/aggregate는 실제 tuples에서 계산하고 table262144/maxleaf2MiB/D512/fixed18/두 journals 제한과 ownership-before-membership을 보존한다. metadata-only 모든 H항목을 무조건 force-read하지 않는다.

Python current/effective/budget/authority/cost/check_call_request/ledger_state는 이 context를 먼저 선택하고 downstream에 같은 four-result ledger_state ABI를 제공한다. Native는 legacy top-level initialization 전에 같은 context를 선택하고 동일 Call-Policy shape와 finite delta 검증을 제공한다. 별도 wrapper가 기존 Native의 재귀 초기화를 그대로 호출하지 않는다. 함수 전체를 복제하지 말고 prospective selection만 필요한 최소 factor한다.

## 실제 generated 경로와 테스트

실제 순서는 다음 하나다: fresh Native register/begin → 실제 ROOT collector/capture Ruff → descriptor-first commit-aware once admin → scanner same ENTRY/fixed_authority_scope/scan_admission/실제 scan_current AST/closed durable report → fixture build/hold/supervise/pre_spawn/pre-Popen/pre-array/production synthetic reader/actual pending/seal → suite build/fixture full-map/CASE/동일 현재 suite receipt/actual pending/seal → 실제 qualification/closed_exit/receipt/Q publication/actual once Q admin/qualification_admin_proof.

현재 prepare/dispatch/capture/collector/admin의 권한·ID·path binding은 context 데이터로 옮기되 명령 whitelist와 root first-FINAL 관측을 유지한다. Native18-digit identity는 JS에서 parse하지 않고 opaque return→Native descriptor-first로 전달한다. admin에 없는 native exit를 LASTEXITCODE로 추측하지 않는다. recorded_once와 role_qualified, known committed와 uncertain commit을 구분하고 오류·불확실 실행은 반복하지 않는다.

build/bootstrap/contract/checkpoint/seal/closed_exit/Q의 현재 numeric pass allowlist는 admitted context로 선택한다. 기존 full Source map/AST/cold fence/fixture/suite/Q 증명을 manifest의 true 값으로 대체하지 않는다. 역사 paid109 P30 Ruff는 actual frozen P30 own9/policy/closed111과 비교한다. 새로운 실제 first-cost는 manifest.start_index111와 실제 newly paid Ruff/cost/actual/request/call/current Source를 비교한다. 합성 paid row나 증거 누락 skip은 없다. 뒤의 begun worker/hold가 있어도 앞서 완료된 Ruff 좁은 증명은 검사할 수 있다.

84 ordered tests/252 phases/current seam/Native15+same PROGRAM16 실제 production primitives/37 extraction slots/one subprocess8초/Source14/full11/late-fourth/nonowned Q·변조 부정 의미를 보존한다. 역사 helper와 새 current helper를 분리하며 과거 expectation을 새 source로 위조하지 않는다. 작은 수정에 새 테스트 캠페인을 붙이지 않는다. 추출/컨텍스트 scaffold 증대가 불가피하면 FIRST에서 실제 semantic site/diff를 ROOT가 검토할 수 있도록 명시한다. AST/import/compile/test/PROGRAM 실행은 지금 제안 단계에서는 NOT_RUN이다.

## 자원·채택·몸체 전환

누적 shared680.0860156/1700/local569.129/1589, 잔여1019.9139844/1019.871을 유지한다. 이번 generated finite schedule caps3/30/30/60/20/40/520/25/40, primary768+same-pending fallback120=888, local 여유131.871은 산술이다. 실제 adapter/setup/collection/closure를 각 role full clock에 포함한다. MAX(call-route,hold-route)/pending substitution/suffix705를 보존하고 fallback은 한번만 예약한다. 새 grant/reset0, 실제 timing/PROGRAM8초는 NOT_RUN이다.

현재 Original admission=false, cumulative10020.115960899504/10200/stage126.361/300/local0/20을 유지한다. 생성 Q만으로 body 권한이나 적합한 비용을 만든 것으로 주장하지 않는다. generated 종료 뒤 ROOT가 실제46–48초 startup와 저장·감사·control300 비용을 바탕으로 충분한 Original amendment 및 science manifest를 작성한다. 이후 이미 리뷰된 startup 보존 유한 ray/E318 asset 대응/neutral identity/후보 동일0→99→300/F1–F8로 계속한다. 최종 보행에는 외부 CPG/teacher/phase/trajectory/root 교정을 쓰지 않는다.

실제 SAME GPT-6.1 Sol/xhigh는 새 prospective 제안만 작성한다. Source4와 필요한 고정 context/Native/caller/collector 제안, 같은 roster 테스트 제안, ABI/필수 site mapping/최소 handoff만 작성한다. canonical/live/GOAL/active plan/history/old FIRST/R1/reviews는 수정하지 않는다. 더 이상의 subdelegation은 하지 않는다. 안정 FIRST 뒤 멈춘다. ROOT FIRST→필요 시 같은 Sol ONE_R1→ROOT SECOND→ROOT residual→actual apply/freeze/admission/genuine 실행을 따른다. P30 sole R1 사용 기록은 유지하며 새 구조 수정의 리뷰 사이클과 혼동하지 않는다. 최신 사람 자율 위임에 따라 별도 승인 질문은 하지 않는다. goal ACTIVE/F1–F8 INCOMPLETE다.
