# BANC M3 E360 P15 — ROOT Source 제안, 2026-10-05

이 제안은 실제 P14의 Native 승인 실패 이후 현재 ROOT가 작성합니다. P15 실행·재승인·추가 grant는 아직 없으며, 현재 P14의 실행 Source·collector·freeze·원장·Ruff 비용을 변경하지 않습니다. 전체 BANC-only finite-goal/F1–F8·해부학 provenance·M2/UI/F6·이전 실패·누적 original 비용을 유지합니다. 이 후속안은 기존 목표의 검증 경로를 복구하며 상위 목표를 새로 만들거나 완료 처리하지 않습니다.

## 실제 P14 결과와 원인

P14는 실제 Astra6/max 설계 검토와 ROOT 반영, 실제 gpt-6.1-sol/xhigh FIRST, ROOT FIRST의 6건 수정 요청, 동일 Sol ONE R1, ROOT SECOND, 실제 Source freeze와 Native pair2 활성화를 거쳤습니다. Ruff는 실제 frozen collector를 채택한 한 await에서 dispatch→첫 genuine FINAL1.840초/exit0, collector proof와 capture proof 모두 true로 통과했습니다. 실제 1.840초를 한 번 청구했습니다. Native capture의 PID33356/Int64 ticks639267587608157543·두 EOF·정상 종료와 현재 matching owned process 없음도 확인했습니다.

scanner request는 생성됐으나 Native callbegin이 line1648의 'Exact unique admitted collector trace'에서 exit1로 거부했습니다. 검사기 dispatch·report·scanner cost는 모두0입니다. Native 프로그램의 [string]$Request CLI 경로와 top-level paid-debit replay의 $request 지역명은 PowerShell에서 같은 변수입니다. line2718의 $request=Checked-Read $call.request가 metadata dictionary를 string으로 강제 변환하고 CLI 경로도 덮어씁니다. Ruff가 하나도 청구되지 않은 최초 실행에서는 이 branch를 지나지 않아 드러나지 않았습니다. 함수만 분리한 Native probe와 Source 파서는 이 실행 범위 충돌을 검증하지 못했습니다.

실제 P14는 CLOSED_ADMISSION_FAILED로 종료했습니다. paid Ruff 자체는 성공이며 live pass.failed_cost=null, failed_cap_s=0, next_index=1을 그대로 보존합니다. 가짜 scanner 반환/실패 cost/clock 또는 Ruff 실패를 만들지 않습니다. closed ledger92/calls42/passes14, pending/reservations/holds/unreturned/original jobs/reserved 모두0입니다. 테스트84×3/252 phases·actual Native test probe·fixture/suite/Q·원본·GPU·BANC 신경·물리·UI는 미실행이며 자연 보행과 F1–F8은 미완료입니다.

근거: verification/banc-natural-gait-m3-20260919/parent-e360-p14-closed-decision.json (6122 bytes, SHA7a3c0f94a36b6f2f29ac3e516733a61ab24f90fed97c9c987af65aa12c0bcfea), parent-e360-p14-closed-ledger92.json (126657 bytes, SHA16e4c4a27da7a358de4786ae75bc5181f4e753a6d8d757d1961e4c4b5120ede4), parent-e360-p14-ROOT-independent-owned-closure.json (69255 bytes, SHA672f2845b17c9607004ebdad8c76723a0d26db672fca51607e2b96b5d0fd9031).

## ROOT가 준비한 구체적인 수정

parent-e360-p14-ROOT-residual-account-proposed.ps1 (317615 bytes, SHA446620862ad61e351ec30c5447166c72cb1caf0e9125af430232c20c668ed2de)은 별도 미채택 Source입니다. 문제가 발생한 top-level branch에서 dictionary를 $p14PriorAdmittedRequest로 읽고 typed mapping을 확인한 뒤 원래 collector proof에 전달합니다. CLI $Request는 파일 경로로 유지합니다. 다른 입력 파라미터·global Describe/JSON-only reader·collector proof·Int64 equality·genuine timing·한도·Source9·실제 기록을 바꾸지 않습니다. Native Source parser0, top-level CLI parameter assignment collision0을 확인했습니다. 이것은 실행 테스트 통과가 아닙니다.

후속 구현은 Native 프로그램 파라미터와 top-level metadata 변수 이름을 구조적으로 구분하고 현재 paid replay branch에 같은 원칙을 적용합니다. ROOT residual은 현재 P14 freeze에 소급 적용하지 않으며 실제 P15 구현은 새로운 명시적 authority와 Source freeze로 진행합니다.

## 목표와 완료 판정

추천 후속 목적은 '성공한 첫 역할을 청구한 뒤에도 그 실제 trace를 재검증하고 다음 역할을 엄격하게 승인할 수 있는 generated 검증 경로'입니다. 완료 판정은 새 승인된 한 전체 체인에서 모든 역할이 genuine cap 안에 종료하고 internal/collector proof, generated fixture, 84개 이름의 3 phases, Q가 모두 실제 통과한 경우입니다. metadata·Source·Ruff 성공은 이 판정을 대신하지 않습니다. 과학적 원본 실행·BANC 실제 보행·F1–F8 완료는 별도 미완료 상태로 유지합니다.

## 새로운 P15 전용 역사·승인 경계

1. P14의 실제 closed-admission 상태를 새 P15 전용 proof로 검증합니다. inherited92의 전체 old91 ordered debits·이전 calls/passes·activation·grant·budget을 보존하고 추가 index91의 actual Ruff once debit, direct request/actual/cost/trace/capture/manifest/두 stream·Int64 ownership·Source freeze를 모두 연결합니다. P14는 failed_cost=null/failed_cap_s=0/next_index1/only Ruff charged를 그대로 요구합니다. 새 proof는 paid-failure predecessor로 오인하거나 기존 paid-failure 함수를 넓히지 않습니다.
2. actual Native callbegin exit1·이미 생성된 scanner request·scanner call 미등록·dispatch/report/cost0·owned closure0을 actual closed-decision/whole92에 결합합니다. predispatch rejection은 paid cost가 아니므로 numeric cap failure를 발명하지 않습니다. 실패 cost 삽입·fake scanner cost·변조된 trace·재작성된 paid clock·없어진 closure·다른 직접 승인은 거부합니다.
3. P15 직접 승인이 있을 때만 새 pair2/event8/grant0를 활성화합니다. 기존 P14를 포함한 activation fields24와 기존 resource grants2를 보존합니다. generated pass15/modified14/total16/original 최대1, 자동 재시도0, P14 재개 false, P16 false, original false를 명시합니다. current callable gate는 P15와 같은 pending의 필수 closure만 허용하며 P14와 이전 pass는 saved completed metadata로 읽습니다.
4. fixed readers는 기존15와 그 정확한 pin을 보존하고 p15_recovery_amendment 한 개를 추가하여16으로 합니다. max_entries16/text_descriptors512/literal262144/aggregate2097152를 넓히지 않습니다. historical relocation46에 실제 P14 실행 Source4의 original/frozen exact same-byte rows4를 붙여50으로 합니다. 현재 Source4 변경은 새 승인/리뷰 이후에만 하며 old owned5와 P13/P14 Native4·collector·전체 실행 증거를 보존합니다.

## 실행 범위와 예산

추가 grant0, 기존 shared1400/local1289/offset42를 유지합니다. 실제 shared583.1700156/local472.213, 잔여816.8299844/816.787입니다. 제안 whole798에 local 산술 여유18.787초이며 속도 보장이 아닙니다. Original cumulative10020.115960899504/stage126.361/E360original0은 그대로입니다.

역할 순서와 caps/suffix는 P14와 같습니다: Ruff3 → scanner20 → fixture build25 → fixture worker55 → fixture seal15 → suite build30 → suite worker600 → suite seal20 → Q30, whole798. suffix795/775/750/695/680/650/50/30/0, fixture future95/suite50, 필요한 같은 pending fallback admin120을 유지합니다. 새로운 역할 admission은 비용·남은 전체 suffix fit을 확인하고 첫 closed failure 또는 Q once debit에서 종료합니다. 그 이후에는 동일 pending failed-worker의 필수 seal/once actual debit/known-owned closure만 허용합니다.

ROOT는 새 frozen collector를 실제 tool body로 채택하고 각 역할을 한 연속 await에서 회수합니다. Date.now dispatch→첫 genuine FINAL, same positive session, 즉시 read30s, opaque raw output, sticky first error, Native separate CreateNew/Flush(true)/strict UTF8/bytes/SHA publication을 유지합니다. 중단·unknown transport·missing FINAL은 hold이며 redispatch/clock reset/child wall 대체가 금지됩니다. frozen own9/collector concrete-fix only/no tenth own source/no hash cycle을 유지합니다. 이번 실제 Ruff collector PASS를 다른 역할의 timely PASS로 일반화하지 않습니다.

## 회귀 검증 설계

기존 named tests84·순서·원래 assertion과 예정252 phases를 유지하고 해당 기존 named test 내부에 paid-replay 회귀를 추가합니다. 기존 Native production fragments15와 단일 actual probe timeout8을 보존하고 정확한 실제 top-level paid replay branch Source를 16번째 paid_replay fragment로 추가하는 변경을 명시적으로 검토·승인 범위에 넣습니다. 추가 Native 별도 실행이나 승인 밖 probe는 없습니다.

새 replay case는 [string]$Request가 실제 CLI 파일 경로를 보유하는 실행 범위에서 실제 paid metadata를 읽어 collector proof까지 호출해야 합니다. 수정된 production branch의 positive, 기존 collision branch의 strict negative, CLI path 값/타입 불변, request mapping 타입, tampered request/trace/clock/cost linkage negatives를 같은 단일 paid Native probe에서 확인합니다. 기능을 새로 흉내 낸 mock proof나 Source 문자열 존재 검사만으로 회귀 PASS를 부르지 않습니다. 원래15 extraction/기존 assertion을 줄이지 않으며 Source-only actual16-fragment assembly exact bytes/dependencies/Native parser를 먼저 검토합니다.

11-source와14-callback full same-pass fixture/suite source equality, duplicate/drift/late-fourth/fail-closed assertions, full suite handoff, generated-only guarded I/O/GPU restrictions, fixed qualification gates를 보존합니다. test 실행은 승인된 suite role 안에서만 수행합니다. actual first predispatch admission failure를 고칠 테스트 case를 추가하되 테스트 이름 수나 단순 Source parser count를 실행 결과로 표시하지 않습니다.

## 실제 리뷰·구현 순서

현재 ROOT 설계 → 실제 GPT-6 Astra/max Source-only 리뷰 → ROOT 반영 → 구체적인 P15 제안에 대한 직접 인간 승인 → 실제 gpt-6.1-sol 적절한 effort FIRST → ROOT FIRST → 필요시 동일 Sol ONE R1 → ROOT SECOND → ROOT-only residual/freeze 순서를 유지합니다. 현재 ROOT residual은 실제 P14 후기 진단을 구체화한 미채택 수정안이며 새 pass 성공이나 새 Sol implementation으로 꾸미지 않습니다.

본 초안은 아직 실제 Astra 후속 리뷰를 받지 않았습니다. Source-only 설계·리뷰 동안 Python/import/AST/compile/Ruff/helper/tests/production Native/probe/원본·GPU·BANC body/neural/UI 실행이 없습니다. 지원 목표 API의 마지막 실제 BLOCKED는 기존 승인 전 상태이며 ACTIVE나 새 목표/예산 reset을 꾸미지 않습니다.

## 승인 근거

현재 docs/BANC_M3_E360_P14_APPROVED_INTEGRATION_20261005.md line5의 'P13 resume·P15·자동 재시도·원본 실행 권한은 false입니다.'에 따라 P15 새 실행을 이번 P14 승인으로 대체할 수 없습니다. 필요한 새 승인은 위 reflected P15 generated whole798 한 번이며 추가 예산/grant 없이 기존 잔여 안에서만 진행합니다. Source와 기록을 먼저 구체화한 뒤 승인 요청을 최종 단계로 남깁니다.
