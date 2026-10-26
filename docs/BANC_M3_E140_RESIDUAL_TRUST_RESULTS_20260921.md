# E140 남은 세 다리의 제약 복구 결과

2026-09-21 · 실행 valid · H140 refuted · M3 pending

현재 세션 설계 → 실제 Sol/high 구현 → 부모 첫 리뷰 → 같은 Sol/high 첫 수정 → 부모 두 번째 리뷰를 완료했다. 새 CUDA 수학이 아닌 기존 신경·기하·sparse solver와 실행 기록의 통합이므로 high를 사용했다. 부모 잔여 소스 수정은 없었다. 생성 시험20개, 독립 수식38항목, 수명주기12항목, 실패 계측4항목이 통과했다. 큰 잔차에서 합산 순서 차이를 오류로 판정하던 검사를 float64 누적 오차 경계로 수정했으며 보행·손실·초기 자세의 합격 기준은 변경하지 않았다.

| 다리 | 실제 응답(시작 재현 포함) | 공동 조건 | 최저 잔차 응답의 손실비 | 같은 응답의 최대 startup 비율 |
|---|---:|---|---:|---:|
| 왼쪽 뒤(LH) | 18 | 미달 | 0.990051757919 | 0.999601155205 |
| 오른쪽 가운데(RM) | 11 | 통과 | 0.988952385178 | 0.999791669931 |
| 오른쪽 뒤(RH) | 18 | 미달 | 0.990001383902 | 0.999600596413 |

손실비는 각 다리의 원래 E128 기준이며 상한0.99다. startup selector 상한0.99980001을 함께 충족해야 한다. 초기 입력3문맥과 원래 변수 bounds를 유지했다. 남은 세 다리의 공동 충족이라는 가설은 이 실행 한도에서 반박됐다. 전체 여섯 다리 묶음은 채택하지 않았다.

원본 CUDA1프로세스/재시도0, optimizer3회, 응답47개. 내부84.324855초/외부88.130209초/감독89.357863초, 보수적 누적1980.357863/2101초다. 원본105개 파일과 dual raw 2428545097바이트를 보존했다. 작은 생성 CUDA20tick은 별도다.

부모는 원자료로 발 기하·근육 출력·손실·모든 변수 미분·초기 제약·CSR 잔차/미분·선택 순서를 독립 검산했다. 3921항목과 241SHA가 통과했다. M2의263파일과 이전1536개 보존 항목도 확인했다. 부모 최초 lifecycle 검증기의 오래된 함수명 오류는 원 기록을 남기고 별도 검증기로 고쳤으며 제품 결함으로 집계하지 않았다. Sol 중간 시험 실패도 보존했다.

이번 단계의 전체158706 BANC·실제 신체 실행은0, F6 노출0, 활성 UI 변경0이다. 이 결과는 조건부 근육 전달 검증이며 자연 보행 완료 판정이 아니다. 다음 결정: Preserve all accepted legs and failed responses. Diagnose remaining active startup/guard rows and the bounded trust trajectory from saved raw before selecting a distinct next method; do not relax selector, objective guard or F1-F8.

근거: `verification/banc-natural-gait-m3-20260919/parent-e140-final-review.json`, `parent-e140-native-output-review.json`, `e140-residual-trust-restoration/`.
