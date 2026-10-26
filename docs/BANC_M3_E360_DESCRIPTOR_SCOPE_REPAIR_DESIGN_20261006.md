# E360 실제 descriptor 한도 실패의 최소 수정 설계 — 2026-10-06

기존 전체 M3 목표와 2026-10-06 실제 body milestone을 계속한다. 필요한 실행 경로만 복구한 후 startup을 보존하는 후보 하나를 동일 인스턴스 control0→99→300의 실제1.5초 감각 폐루프로 평가한다. 고정 F1–F8, anatomy/provenance, M2/UI 및 모든 실패는 보존한다. 추가 승인 질문은 하지 않는다. 이 문서는 ROOT 설계이며 구현 또는 실행 성공이 아니다.

## 실제 관측과 실패 종료

P25 ROOT 실제 FIRST/SECOND 후 Source4/own9/Native4/current policy를 적용하고 단일 Native activation exit0를 관측했다. 기존105개 debit/55calls/23paid passes를 보존하고 pair2/grant0만 추가했다. 실패한 P24의 Source와 실행 기록은 원형 보존했다.

P25 Ruff는 actual full-return1.695s/exit0/cap3, scanner는4.716s/exit1/cap30이며 각각 한 번 차감했다. 두 비용 모두 collector proof와 administrative internal validity가 true인 실제 실행 기록이다. scanner exit1과 Q 미수행을 성공으로 해석하지 않는다. 실제 traceback은 main→effective_policy→current_scope_authority→completed_historical_p23/p22/p21_proof→historical_p12_stream_bytes:1726의 `same stream descriptor bound`다. 구조 scan 본체, fixture/suite/Q, 원본 신체/신경/CUDA/UI에는 진입하지 않았다.

실제 closed107은199728B/SHA097201fd2c51b8445d0415beea88248713233f127e581c5e3c7233ec9db66f43이다. 비용107/호출57/paid passes24, P25 next2/failed scanner cap30, all charged, pending/hold/reservation/original ownership0이다. shared672.5270156000001/local561.57, 기존 한도1700/1589, local 잔여1027.43으로 whole888을 수용한다. 새 grant/reset은0/false다. 원본 cumulative10020.115960899504/10200, stage126.361/300, E360 local0/20는 그대로다. 원본 실행 한도는 실제 body 전환 전에 별도로 충분하게 등록해야 하며 generated 예산을 원본 예산으로 대체하지 않는다.

## 원인 가설과 최소 변경

Source상 FixedAuthorityScope._consumed는 depth0에서 등록을 하지 않는다(287–308). fixed18 reader의 validator가 활성인 depth>0에서 소비한 의존성만 parsed-literal manifest에 등록하며, manifest512/parsed18/aggregate2MiB/cold fence/poison/ENTRY를 유지한다. 반면 historical_p12_stream_bytes는 positive stream을 _consumed로 처리하면서 empty stream은 depth와 관계없이 scope.descriptors에 직접 등록한다(1692–1731). 실제 실패 stack에는 fixed-reader validator frame이 없고, scanner main8216의 outer scope에서 current selector가 직접 whole historical proof를 실행한다. 이것은 empty stream만 fixed contract manifest를 오염시키는 depth0 경로와 일치한다. 실제 depth 값과 전체 소모 path 수를 별도 측정한 것은 아니다.

추천하는 첫 수정은 **고정 literal 의존성 등록과 직접 실행별 역사 검증의 소유 범위를 일치시키는 최소 수정**이다. empty history stream의 manifest 등록도 depth>0일 때만 한다. 단순히512를 키우거나 scope를 쪼개어 우회하지 않는다. exact completed allowlist/bytes0/SHA-empty/contained path/strictUTF8/two stable reads/ADMIN_CHECK/scope clock/failed-scope poison을 유지한다. depth0에서도 실제 uncached 역사 읽기와 내용 검증은 수행한다. parsed history, current selector, budget, fence 성공을 cache/stub하지 않는다.

Astra는 이 변경이 late historical drift와 실제 publication/pre-spawn/pre-array fence를 약화시키지 않는지 반드시 검토한다. 같은 기존 positive stream의 depth0 의미를 근거로 사용할 수 있으나 기존 empty stream의 안전 요구를 자동으로 없애지 않는다. Source상 이 방법이 안전하지 않거나 관측 실패를 해소하지 못한다면, 가장 작은 안전한 대안과 필요한 계약 변경을 명시한다. 불명확한 전체 verifier 재설계, cap 증액, 별도 private preflight로 확대하지 않는다. 통과한 확인을 새 구체 문제 없이 반복하지 않는다.

## 검증과 실행 방법

ordered84 이름/252phases/fixed18/descriptor512/ENTRY-depth0-poison/full11/child Source14/late fourth/nonown Q/current512KiB/generic256KiB/CASE-Q64KiB/receipt34-old33 및 원래 Native15/same PROGRAM16/one subprocess8은 유지한다. 기존의 `test_fixed_scope_entry_literal_aggregate_and_monotone_descriptor_bounds` 또는 적절한 기존 이름 안에 실제 production historical stream reader의 depth0/depth>0/fixed512-full/duplicate/unsafe empty/failed scope/late drift를 연결한다. 새로운 테스트 함수나 구현의 거울 predicate를 만들지 않는다. 기존 actual hot full current ledger arity4/_totals/_local와 실제 PROGRAM16 positive/Boolean consumer를 유지한다.

Source 단계에서는 Native text/JSON/SHA/regex/ParseFile만 허용한다. Python/import/AST/compile/Ruff/test/assembled PROGRAM/private probe/preflight/원본 array/header/hash/checkpoint/CUDA/신체/UI는 실행하지 않는다. 실제 freeze/activation 후 ROOT가 기존 순서의 실제 role을 한 번씩 수행하고, 첫 closed failure를 보존하여 종료한다. fixture/suite/Q가 유효하고 새 구체 문제가 없으면 실제 body milestone으로 넘긴다. source/lint/qualification 성공은 보행 성공이 아니다.

## 현재 method와 attempt data의 경계

후속 현재 namespace는 P26이지만 본질은 위의 dependency ownership 변경이다. 번호/고정 predicate count/파일 digest만 바꾼 재시도는 허용하지 않는다. 변경된 방법과 이전 실제 P25 scanner 실패를 연결한다. 모든 실행별 역사 사실은 immutable frozen P25/actual closed107/actual Ruff8+scanner4와 정확한 Source9/Native4/collector로 검증한다. 기존 고정 계약과 현재 attempt data를 helper에서 명확히 분리하고, 오류 메시지의 낡은 count/version을 authoritative 데이터처럼 사용하지 않는다. 이 단계에서 전체 historical validator를 새 프레임워크로 바꾸지 않는다.

paid index104=P23,105–106=P25,107이후=currentP26이며 P24 paid range는 없다. 현재 namespace26과 completed paid passes24를 혼동하지 않는다. activation은107/57/24를 유지하며 fresh Ruff 등록으로25passes/58calls, 실제 첫 charge로108costs가 된다. fresh 이전 실패는 P25 scanner30/next2다. 현재 real activation predicate는 count107, 기존 positive는 real closed107/real current freeze/authority/locktrue/Booleantrue와 실제 first comparator END를 확인한다. count106/lock-held/current-pair negatives는 normal Booleanfalse/reached/null failure를 소비한다. future 거부값은27이다. Source 전임자와 paid 전임자는 이번에는 모두 실제 frozen P25다. 실패한 unpaid P24는 별도 불변 보존 대상이다.

ROOT apply 최종 보존은 protected322, prior oldNative64+actualP25 Native4/collector5=oldNative69, P25 Source4/frozen4/current policy/freeze/reviews/executed operations/costs/trace/snapshot/decision 및 모든 prior artifacts다. 기존 root GOAL151552/SHA16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace와 actual human autonomy1098/SHA8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224를 보존한다.

roles9/caps3,30,30,60,20,40,520,25,40; primary768/admin155/fallback120/whole888; primary suffix765,735,705,645,625,585,65,40,0; fallback885,855,825,765,745,705,185,160,120; future125/65/MAX825/705; fixture60/55/51/42/suite520/480/450/360/host2GiB/output16MiB를 유지한다. 첫 FINAL/continuous same-session/opaque raw output/Native Int64 ticks/owned close/once debit도 유지한다.

ROOT 설계→실제 GPT-6 Astra/max 리뷰→ROOT 반영→실제 GPT-6.1 Sol의 난도에 맞는 effort FIRST→ROOT FIRST→필요시 같은 Sol sole ONE_R1→ROOT SECOND→ROOT 잔여 수정→actual freeze/activation/runtime 순서다. 목표는 ACTIVE/INCOMPLETE다.
