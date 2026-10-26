# E360 P22: 과거 정책 descriptor 비교 수정

## 달성할 결과

실제 실패한 P21은 그대로 보존한다. P22의 실제 scanner가 과거 정책의 원래 논리 descriptor를 현재 정책 파일과 혼동하지 않고, 고정 Source 검증 조건을 모두 만족하는 closure를 생성하도록 수정한다. 기록 경로의 이전 pass 상태 키도 새 pass의 실제 키와 일치시킨다. 코드 검사 통과, 완료 metadata의 구조적 유효성, Source closure 통과, 생성 fixture/suite/Q 통과, 실제 BANC 보행은 서로 다른 결과로 기록한다. 전체 자연 보행 목표와 F1–F8은 ACTIVE/미완료다.

## 확인된 근거와 기준점

- P21 Ruff: 실제 dispatch→first FINAL 1.605초/exit0/cap3, once debit. Scanner: 6.868초/exit1/cap30, once debit. 두 비용의 collector와 completion metadata는 유효하지만 scanner 실행은 실패다. P21 총 실제 비용8.473초. fixture/suite/Q와 original/body/neural/GPU/UI는 NOT_RUN이다.
- 실제 scanner traceback은 `ledger_state` line5487의 `historical/current cost amendment.bytes`에서 종료됐다. line5486은 index95 이후 `CURRENT_REPAIR_POLICY`로 경로를 고르고, line5487은96..98/P18만 별도 descriptor로 비교한다. ROOT가 실제 index99 P19 비용의 원래6880-byte descriptor와 현재7781-byte descriptor를 독립적으로 읽어 불일치를 확인했다. traceback 자체에 index가 찍힌 것으로 주장하지 않는다.
- `verification/banc-natural-gait-m3-20260919/parent-e360-p21-closed-ledger103.json`:178477 bytes/SHA256 `23886a8fbcc12d0db312d2721c3a568c04e2f015682f00ce729563743a3f8c07`. 비용103/호출53/pass21, 모두 charged, pending/reservation/hold/unreturned/original job0.
- `parent-e360-p21-ROOT-closed-decision.json`:5256 bytes/SHA256 `dd1d55be4a0f30d06c9c0d2c168d4756ca374225d87659379ec046820fc6303b`. 실제 shared/local662.4540156/551.497, 한도1700/1589. 남은 local1037.503에 whole888이 들어가고 여유149.503이다. 새 grant0, reset 없음, 실제 P19 grant300을 포함한 기존 grant3 그대로다.
- 실제 P21 frozen policy7781/SHA `c4bf79de3e8efdf90170916eb6ea24747aa1830e4799885e06dc019614b4dda5`, finalfreeze12292/SHA `556486c5ce50b00dc8ffce6230f009adee21495878564c048d9cd18cfbbc34b8`, Source4/own9/Native4/collector와 원래 논리 경로를 보존한다. P21 Ruff와 scanner는 각각 index101,102다.
- 초기 Source-only 수집기의 잘못된 literal은 Sol sole R1으로 고쳤고 초기 freeze/리뷰도 보존했다. 실제 Native bookkeeping line58은 여전히 `e360:p20:ROOT-genuine:`를 기대했다. ROOT residual의 한 문자열/한 호출 경로 수정으로 원본 FINAL을 바꾸지 않고 두 비용을 출판·once-account했다. 이 ROOT 출판 성공을 기존 canonical trace branch의 성공으로 해석하지 않는다. 해당 branch 및 `parent-e360-p21-ROOT-residual-publication-review.json`3521/SHA `1fdf5e2d4d9cd320681291222b778f3fa82dcd86acb6446b9dbf97944f455186`도 보존한다.

## 수정 범위

1. Python의 비용 amendment 비교를 단일한 명시적 typed descriptor 선택 경로로 모은다. 기존59..95 정책 경계를 유지하고96..98은 P18,99는 P19,100은 P20,101..102는 P21의 원래 논리 descriptor를 반환한다. current P22는103부터 현재 compiled policy descriptor만 사용한다. 과거 descriptor의 논리 경로는 그대로 비교하며, 과거 bytes를 읽을 때는 정확한 physical frozen policy/Source만 읽는다. index99/100의 뒤쪽 추가 검사만 통과시키는 방식으로 앞쪽 generic check를 남겨두지 않는다. 현재 admission/actual/call-policy/concrete-fix/collector/worker/receipt/Q에서도 동일한 역사·현재 경계를 확인한다.
2. P21의 실제 완료 이력은 한 direct metadata proof로 처리한다. immutable closed103, 기존103개 비용, 기존 모든 call/pass/activation/grant와 original counters, P21 pair2/grant0, 실제 own9/Source4/Native4/collector, original logical policy, 실제 Ruff1.605/exit0/cap3와 scanner6.868/exit1/cap30, next2/failed scanner cost, first FINAL와 once charge를 검증한다. schema가 없는 실제 기록에 schema/pass_id를 발명하지 않는다. Source4의 현재 논리 경로를 P21의 같은-byte physical frozen copy로 정확히 옮겨 읽는다. Frozen P21 모듈을 import/clone하지 않는다.
3. 현재 P22 namespace를 Source4, ordered Native4, ROOT collector, current policy, bootstrap, helpers, Native capture/dispatcher/worker/receipt/Q까지 연결한다. Native 비용 선택은 기존 역사 그대로이며101..102는 P21 original descriptor,103부터 P22다. 실제 fresh-pass guard는 generated predecessor21개와 P21 scanner once cost/cap30/next2를 요구한다. P21·과거·미래·original의 dispatch/reopen은 거부한다. 과거 grant3 budget proof는 current pair와 독립적이고 재귀하지 않는다.
4. 새 P22 canonical bookkeeping의 실제 trace publication branch가 P22 schema/pass/call/state key를 검사하게 한다. Collector const literal도 새 P22 compact policy의 실제 bytes/SHA와 일치시킨다. 예외를 허용하거나 검사 조건을 줄이지 않는다. ROOT residual의 private publication adapter로 canonical 성공을 대신 주장하지 않는다.
5. 기존 ordered84 함수와252 phases 안에서 실제 production 비용 선택과 trace gate를 회귀 검증한다. 과거 P19/P20/P21 original logical descriptors/current slot 대체 거부, index99/100/101/102 경계, current103 경계, 타입/byte/SHA/path 변조를 다룬다. 이미 있는 actual whole-history positive를 재사용하고 negative마다 whole history를 재실행하지 않는다. 실제 Native trace guard의 complete current positive와 old/partial/mismatched key negative는 같은 PROGRAM16/기존 process boundary에 넣는다. 새 역사 fixture layer, reader, 별도 subprocess/test suite, 별도 lint preflight를 만들지 않는다.

## 진행 순서와 적용 경계

ROOT 설계 → 실제 GPT-6 Astra/max 리뷰 → ROOT 실제 반영 → GPT-6.1 Sol 구현(연결·회귀가 복잡하므로 xhigh) FIRST → ROOT FIRST → 필요시 같은 Sol sole ONE_R1 → ROOT SECOND → 남은 수정은 ROOT → 보호 대상과 proposal의 최종 audit/apply/freeze → 실제 pair2/grant0 activation → fresh genuine validation. 이 계획 자체는 새 정책·activation·runtime 승인 증명이 아니다. 모든 리뷰·수정·Source pins를 실제 파일로 남긴다. Source proposal에서 Python import/AST/compile/Ruff/tests/assembled PROGRAM/target gate를 실행하지 않는다. 첫 검사 역할은 고정한 Source를 사용하는 실제 Ruff3다.

P21 as-executed 및 initial pre-runtime freeze, Source4/own5/Native4/collector/policy/실제 반환·잔여 출판 수정·old Native histories를 보존한다. 기존 보존 inventory는 P21 baseline protected322/oldNative44에 실제 P21 Native4/collector를 추가한 oldNative49가 된다. 실제 role Source와 ROOT-only metadata utility를 구별한다. `GOAL_PLAN.md`151552/SHA `16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace`는 수정하지 않는다. 새 current pair2만 허용하며 전체 closed103의 모든 old 필드는 typed-equal, ownership0/lock 없음/whole888 적합/실제 reviewed freeze 후에만 commit한다. 추가 승인 요청 없이 기존 사용자 위임 범위에서 결정한다.

## 고정 검증·자원 조건

역할9개는 Ruff/scanner/fixture_build/fixture_worker/fixture_seal/suite_build/suite_worker/suite_seal/qualification이며 cap3/30/30/60/20/40/520/25/40초다. primary768/admin155, same-pending fallback120 별도, whole888. primary suffix765/735/705/645/625/585/65/40/0, unused fallback 포함885/855/825/765/745/705/185/160/120이다. fixture60/55/51/42, suite520/480/450/360, host2GiB/output16MiB, primary future125/65와 fallback120을 별도 검증한다.

고정 literal reader18·descriptor512·clock/poison, current contract/bootstrap512KiB, generic/preSpawn/outer/fullExit256KiB, CASE/Q64KiB, receipt34/old fields33·실제 ENTRY/callable/depth0, original Native15와 같은 PROGRAM16/subprocess8, full11/child Source14/late-fourth nonown·fail-closed Q를 유지한다. 정확히 등록된 과거 zero-byte stderr 예외만 유지하고 generic positive Source 조건을 넓히지 않는다. 모든 역할은 실제 dispatch→처음 FINAL을 즉시 수집해 opaque 출력을 보존하고, 완료하지 않은 호출을 재전송하지 않으며 실제 비용만 한 번 차감한다. 첫 closed failure나 Q once debit 후 새 P22 역할을 종료하고 ownership을 닫는다.

## 완료 판정과 이후 목표

이번 수정은 새 P22 Ruff·scanner·고정 생성 fixture/suite·Q의 실제 결과로 판단한다. 승인된 기준을 바꾸어 통과시키지 않으며 실패 후보·시간·소스·노출을 보존한다. genuine Q와 original 계약/자원·소유 조건을 실제로 충족한 다음에만 같은 제어 경계의 BANC effective motor96/관절42/힘줄6·접촉·지지·미끄러짐 관찰을 계획한다. 이 단계의 Source/metadata 성공만으로 자연스러운 보행이나 생물학적 기전, F1–F8, 전체 목표 완료를 주장하지 않는다.
