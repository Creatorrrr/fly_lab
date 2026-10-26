# E170 활동 이력 전달 적합 결과

새 적합기는 구현·리뷰·원본 실행을 마쳤으나 **후보는 기각**했다. 조건부 관절 명령 오차 감소는4.6363%로 목표20%에 미달했고, startup과 일부 근육 보호 조건도 통과하지 못했다. 전체 M3 자연 보행 F1–F8은 미완료다. 원본 결과는 유효한 제한 가설 시험이며 실제 몸체 보행 개선이나 성체 다리의 생물학적 기전 검증은 아니다.

## 구현과 검토

부모가 Stage B V2를 설계하고 실제 `gpt-6-sol/high` `/root/e170_gpt6sol_history_fit`가 첫 구현 및 FIRST 수정했다. 부모 FIRST 리뷰에서는 후속 native/affine 검산 예외 시 이미 계산한 candidate가 유실되는 문제를 생성 자료로 재현했다. 같은 Sol이 caller-owned raw 보존을 고쳤고 부모 SECOND 리뷰에서 재현이 해결됐다. 남은 기본 Ruff3건은 부모가 수정했다.

최종 파일은 `tools/fit_c_banc_m3_history_recruitment.py` (e6acf34cf20bd175ee3944f8f3e0f74656fc7d095b56c200db2d14668e9b23e3), `tests/test_c_banc_history_recruitment_fit.py` (66255c4cc694b3bb0e7005ffb392313c2bed13124702cbc26c268120d63a447e)다. 생성 시험6개 및 수정 회귀2개, 부모 독립 수치187개를 확인했다. 수치식은 해당 검산 snapshot과 동일하며 원형 resource 검사만 feature 구성 이전으로 옮겼다. 부모 기본 Ruff 검사/format도 통과했다. 작성자의 최초 Ruff passed 기록과 달리 부모 기본 설정에서3건이 검출됐다는 이력을 보존했다.

산술 진단에서는 시간축 고급 인덱싱의 메모리 배치가 축약 순서를 바꾸는 문제를 확인했다. C-contiguous 곱의 tick별 축약과 GEMV를 사용해 기준값을 완화하지 않고 원본 후보의 native activation/command/velocity 오차를 모두0으로 만들었다. tick0는 첫 decode 전 저장 명령이고 canonical decoder q0와0.32668292642150043만큼 달랐다. V1을 보존하고 V2에서 실제 decode1..2000을 비교했으며 target100..1923,velocity1823쌍,startup1..99와 허용치는 유지했다. q0 어느 값도 덮어쓰지 않았다.

첫 metadata-only preflight는 부모 gate schema를v1로 잘못 기재해 거부됐다. 원본 배열/fit0인 상태에서 실패 기록을 보존하고 실제 source의v2와 일치하는 별도 gate를 만들었다. 최종 preflight는3584개 frozen hash를 검증했다.

## 원본 한 번의 결과

| 항목 | 기존 E150 | E170 반환 후보 | 요구 |
| --- | ---: | ---: | ---: |
| 전체 command RMS | 0.05668638535 | 0.05405821481 | 기존의80% 이하 |
| 전체 velocity RMS | 1.94664326116 | 1.86807121687 | 기존 이하 |
| startup q RMS | 0.01772709268 | 0.02145682470 | 0.018 이하 |
| startup q 최대절대오차 | 0.03945779799 | 0.04768183270 | 0.04 이하 |

여섯 다리 각각의 command/velocity RMS는 감소했지만 전체 채택 조건을 만족하지 않았다. alpha/beta exact box는 통과했다. 실제 MN floor의 최저 여유는−2.918279054363666e−10, 근육 upper 최대는2.0000000004184284여서 기존1e−12 물리 허용치를 넘었다. 계수를 clipping하거나 임계값을 바꾸지 않았다.

Clarabel은 `AlmostSolved`, margin−0.1920458165451506을 반환했다. 단순 상태 문자열을 성공 근거로 사용하지 않았다. CPU1, solver14.87834499998초, 전체 supervisor72.49079279997초, 출력113577007bytes로 사전 cap 안에서 종료됐다. 원본 process/fit/return 각각1, history update2000, 새 신경·물리·몸체·store·EMA 실행0이다. UI/M2/F6는 변경하지 않았다.

## 독립 검산과 해석

부모는 저장 raw의 scalar activation/명령/물리 합산/지표,24642 선형행 및15SOC/18QR를110개 항목으로 검산했다. 최대 scalar activation 차이4.2633e−14, command2.3315e−15였다. 실제 native와 candidate 비교의 activation/command/velocity 차이는 각각0이다.

별도의 실제 `gpt-6-sol/high` 감사는 sparse A(44443,5941), projected dual, strict inner box와 실제 physical box를 독립 합산했다. solver의1e−9 coefficient 하한은 RHS와 stationarity box 양쪽에서 제거하고 physical floor/budget1e−12를 포함했다. 실제 물리 범위를 포함한 margin 상한은 **−0.19204579859728288**로 여전히 음수였다. 따라서 저장된 E150 신호·현재 alpha+F*beta 가족·기존 startup/velocity/물리 제약을 함께 만족하는 affine 조건부20% 목표는 배제된다. 이는 유한정밀도 저장 모델의 결론이며 다른 신경 동역학/전달 구조/실제 폐루프 보행 또는 자연 보행 전체를 반증하지 않는다.

E169 원본의 technical invalidity는 그대로다. 이번에는 native 계산 결함을 해결해 유효하게 평가했지만, 개선 목표가 통과한 것은 아니다. 이 기각을 전체 M3 완료로 바꾸지 않는다.

## 증거와 다음 결정

증거는 `verification/banc-natural-gait-m3-20260919/`의 `e170-history-recruitment/`, `e170-history-recruitment-supervision/`, `e170-parent-saved-verification/result.json`, `parent-e170-saved-certificate-audit.json`, `parent-e170-final-source-review-v2.json`에 보존했다. 부모 saved 검산3.2761초, 별도 certificate 감사0.7050초는 original process 비용과 구분한다. 이전 비용과 산술 진단을 포함한 보수 original 누적은3045.490792799974/3093초다.

같은 양수 전달 계수나 시간상수의 반복 탐색은 선택하지 않는다. 다음은 현재 신경 특징의 표현력과 물리/startup 제약의 병목을 저장 QR로 분리하는 작은 진단이다. 이 판별 후 기존 해부학 안의 신경 동역학 또는 근육 전달 구조 중 변경할 위치를 결정한다. 그 계획은 `docs/BANC_M3_AFTER_E170_DIRECTION_REVIEW_20260923.md`에 기록하며 새 원본 적합은 아직 배정하지 않았다.
