# BANC M3 H 실제 검증 결과 (2026-10-08)

H는 NOT_QUALIFIED입니다. 실제 lint 실행은 suspended Job의 begin 기록 이후 resume 전에 같은 ENTRY의 10초 child 제한을 초과해 exit124로 종료됐습니다. Ruff 실행은 시작되지 않았고 두 출력은 빈 값입니다. 실제 collector 전체 반환 소요는 13.253초이며, Native once commit으로 정확히 한 번 기록했습니다.

ROOT가 기존 285개 debit/167개 call을 포함한 전체 history와 순서대로 계산한 binary64 총계의 보존을 독립 확인했습니다. 현재 286 debits/168 calls, 모든 12개 owner/reservation/pending/HOLD 맵은 비어 있고 실제 Job/process 종료와 두 EOF가 확인됐습니다. 공통 lock도 없습니다. 남은 Direct Generated596.1470000000002초, shared603.9349843999998초, Original3336.9150391004946초입니다. grant/refund는0입니다.

앞선 실제 Astra/max 설계 리뷰와 GPT-6.1-Sol/max FIRST·유일한 R1·ROOT 두 리뷰를 완료하고 코드를 적용했습니다. R1은 생산 경로의 H 4-cost 검사를 고쳤으며 349개 기존 전체 검증문과 FIRST375개를 보존했습니다. 이 Source 리뷰 결과는 lint나 실행 성공을 뜻하지 않습니다. H math/gate-a/gate-b/Original body/audit는 NOT_RUN이며 실패로 차단돼 있습니다. 현재 자연스러운 BANC 보행은 검증되지 않았고 F1–F8, M2/UI, candidate 상태는 그대로입니다.

Source에서 transition이 current_caps를 호출한 직후 generated_budget_limit이 같은 current_caps를 다시 호출하는 반복을 확인했습니다. 이를 포함한 전체 사전 검증 비용이 지연에 기여한다는 가설을 다음 설계에서 다룹니다. 함수별 실제 시간은 측정하지 않았습니다. 기존 실패 기록과 고정 Source를 보존하고 남은 예산 안의 별도 새 전체 검증으로만 개선을 확인합니다.

전체 근거: verification/banc-natural-gait-m3-20260919/parent-h-qualification-lint-repair-ROOT-actual-result-20261008.json (11722 bytes / 1f4ecc5a458ef1ccb731423c92029dd846ffb656a373d8c62eb8db519f88cdc6); parent-h-lint-closed286-20261008.json (458157 bytes / 0c5b10a4e68bcc5bbe8b485bce872c3de219b9a9af2f7992a010c831eff25b98).

