# E274 FIRST 리뷰 — 첫 수정 요청

2026-09-27 root. FIRST 3개 source는 `parent-e274-first-review.json`과 `parent-e274-first-*.txt`에 동결했다. E270 runner 변경이 설계한 함수 주입 두 곳뿐임을 독립 비교했다. LP 부호, triple/box/radius, 실제 startup 판정, train-only 입력, 기존 최종 기준은 설계와 일치한다. Sol 생성 CPU43개 통과 보고도 확인했다.

1. **예산 종료와 저장 오류가 동시에 발생하면 오류가 숨겨진다.** `fit_feasible_leg`는 BudgetExceeded를 primary로 보존한 뒤 `persist_all` 실패를 note에 넣지만 그 예외를 다시 올리지 않고 selected를 반환한다. root 생성 재현은 실제 LP/GPU 없이 평가 2개와 stub 방향 1개로 완료 후보가 반환됐지만 `selected.npz`와 `selection.json`이 없는 상태를 확인했다. 과학 결과와 비용 기록을 저장하지 못했으므로 정상적인 bounded 종료로 계속할 수 없다. 저장이 성공한 budget-stop은 완료 후보를 반환하고, 저장 실패 시에는 원래 BudgetExceeded와 그 secondary note를 상위 경계로 보내 원본을 중단해야 한다. 원래 계산 오류도 계속 동일 예외로 보존한다.

2. **시간 한도 직후 완료된 평가의 proposal 이력이 미완료로 남는다.** 기존 evaluator는 평가를 cache/evaluations에 먼저 저장한 뒤 budget check로 예외를 올릴 수 있다. 새 optimizer는 그 순간 proposal의 evaluation/status/margin을 기록하지 못한다. 해당 계수의 cache에 완료 항목이 있으면 완료된 evaluation 번호와 원래 수치, budget scope를 기록하고, 다음 iterate로 채택하기 전 종료됐음을 표시한다. startup만 완료했거나 평가 전체가 미완료면 그렇게 구분한다. 최종 choose_candidate의 완료 strict-feasible 후보 보존 규칙은 바꾸지 않는다. 별도의 실제 평가를 다시 수행하지 않는다.

같은 실제 gpt-6-sol/high가 두 문제의 첫 수정을 하고 관련 생성 CPU 회귀를 추가한다. 새로운 LP 탐색/수학/허용치/원본 실행은 추가하지 않는다. root SECOND 후 남은 문제가 있으면 root가 처리한다.

E273 최종 gate는 `verification/banc-natural-gait-m3-20260919/parent-e273-final-review.json`이며 SHA는 `54d89ea5251a11f586b4a81bb4871cf3034f9519b7f9c12d3296476a1635a304`다. 동일 입력 memory-matched 생성 비교의 median 2.722835배, 36개 bit-exact 비교를 통과했다. FIRST 성능 실패는 별도로 보존했다. 이 문서 시점 E274 원본 예약은 여전히 0이다.
