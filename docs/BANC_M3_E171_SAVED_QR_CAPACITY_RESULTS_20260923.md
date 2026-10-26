# E171 저장 QR 표현력 진단 결과

E171 V2는 저장된 static/history affine 행렬에12회 고정 SVD를 수행했다. 실행과 산술 검산은 유효하지만, 사전 수치 해상도 조건을 두 가족 모두 충족하지 못해 결론은 **inconclusive_numerical_resolution**이다. 후보는 채택하지 않았고 BANC 자연 보행 F1–F8은 계속 미완료다.

| 저장 affine 진단 | Static | History |
| --- | ---: | ---: |
| baseline command RMS | 0.05668638535 | 0.05668638535 |
| 반환 command RMS | 0.03175462610 | 0.02627596754 |
| 명목 오차 감소 | 43.9819% | 53.6468% |
| 수치 해상도 지표 RMS | 0.00330838450 | 0.00792031087 |
| 허용 지표: baseline의0.1% | 0.00005668639 | 0.00005668639 |
| 계수 최대절대값 | 316,313,137.91 | 118,426,137.17 |
| startup RMS, baseline0.01772709 | 227,199.48 | 3,399,572.05 |
| velocity RMS, baseline1.94664326 | 1.68754826 | 1.31589307 |
| MN floor 위반 /330 | 309 | 303 |
| 근육 lower/upper 위반 /77 | 76 /46 | 77 /74 |

위 수치는 실제 몸체에 명령을 적용한 보행 결과가 아니다. 부호/크기 제한을 해제한 저장 행렬의 계산값이며, 극단적 상쇄와 시작 구간 확대가 나타났다. 원래 계수 한도2 및 근육 예산을 충족하지 못한다. QR에서 시간별 원래 최대오차를 복원하지 않았으므로 startup pointwise max, 실제 activation/native 동작 및 실제 신체 안정성은 평가하지 않았다.

정해진 cutoff에서 static rank는 LF164/354, LM146/312, LH122/318, RF158/360, RM159/324, RH155/312였다. history는 LF328/708, LM291/624, LH244/636, RF316/720, RM316/648, RH310/624다. 작은 singular direction을 포함한 일부 retained condition은1e12–1e13 규모다. 이 rank와 잔차를 전체 BANC 또는 무제한 함수 가족의 불가능성 증명으로 해석하지 않는다. 해상도 지표는 사전에 정한 보수적 heuristic이며 엄밀한 수치 오차 인증도 아니다. 실제 독립 scalar/vector 잔차 차이의 최대 L2는 static6.03e-12/history2.68e-6이었다. 이 추가 검산을 이유로 사전 분류 기준을 사후 완화하지 않았다.

첫 V1 원본 호출은 부모 설계가 메모리 내부 `pair_start/pair_end`를 저장 파일 필드로 혼동하여 실패했다. 실제 저장 필드는 `affine_rows_velocity_start/end`였고 process1/fit0, 원래 whitelist 첫5개 배열만 읽고 QR에 도달하지 않았다. V1 결과·소스·freeze·gate는 보존했다. 새 적합 전에 V2에서 필드 alias와 ZIP metadata 확인을 명시적으로 동결했으며, cutoff/수치 기준/목표 구간/생물학 기준은 변경하지 않았다. V2 과정의17시험과 전체 생성 실행이 통과한 뒤 별도 gate로 한 번 실행했다.

개발은 실제 gpt-6-sol/high의 FIRST 구현→부모 FIRST 리뷰→같은 Sol 첫 수정→부모 SECOND 리뷰→부모 잔여 수정 절차를 따랐다. 부모 생성 독립 검산965개, 실제 preflight를 포함한 생성 실행25개 검사 및 원본 저장 결과의 독립 검산2742개가 통과했다. 첫 raw 저장 실패에서 반환값 보존, 잘못된 저장 schema를 배열 열람 전에 차단하는 경로도 검증했다. 기존 E169 원본 invalid와 E170 조건부 기각은 변경하지 않았다.

V1 감독0.5126642초와 V2 감독3.0259616초를 이전 누적에 더한 보수 original 누적은3049.0294185999082/3167초다. V2 내부2.718초, 출력42,999,205bytes, CPU/BLAS1, process1/fit12/자동 retry0이었다. 별도 저장 검산4.3948962초는 원본 계산과 구분해 기록한다. 새 신경/EMA/history/물리/몸체 실행0, F6/UI/M2 변경0이다.

관련 증거:

- [V1 실패 판정](../verification/banc-natural-gait-m3-20260919/parent-e171-v1-failure-review.json)
- [V2 source gate](../verification/banc-natural-gait-m3-20260919/parent-e171-final-source-review-v2.json)
- [원본 V2 결과](../verification/banc-natural-gait-m3-20260919/e171-saved-qr-capacity-v2/result.json)
- [독립 저장 검산](../verification/banc-natural-gait-m3-20260919/e171-parent-saved-verification/result.json)

다음 선택은 [E171 이후 계획](BANC_M3_AFTER_E171_DIRECTION_REVIEW_20260923.md)에 기록한다. 무제한 계수의 낮은 진단 오차를 실제 개선으로 채택하거나 같은 E171 gate로 추가 fit을 실행하지 않는다.
