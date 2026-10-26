# E174: 시작 응답 보호가 고정 static 전달의 개선을 제한한다

E174의 원본 두 비교는 valid_diagnostic으로 완료됐다. **같은 저장 static 신호에서는 startup 보호만 적용해도 기존 command RMS 20% 감소 목표가 배제됐다.** 독립 검산한 true 영역 margin 상한은 −0.19342340537646155다. 이는 현재의 고정 양수 전달계수 가족과 저장 궤적에 대한 결과이며, 전체 BANC가 걸을 수 없다는 증거가 아니다.

근육 보호만 적용한 비교는 **inconclusive**다. 명령 오차는 23.0852% 감소했지만 반환 계수가 기존 MN/근육 보호 허용오차를 약간 넘어 직접 witness로 인정하지 않았다. 양수 dual 상한도 배제 증거가 아니므로 가능·불가능 어느 쪽도 확정하지 않는다. 기준을 완화하거나 재시도하지 않았고 보행 후보로 채택하지 않았다. 전체 자연 보행 F1–F8은 계속 미완료다.

| 반환점의 항목 | baseline | startup 보호 | 근육 보호 |
|---|---:|---:|---:|
| command RMS | 0.05668638535 | 0.05412069420 | 0.04360020793 |
| command 오차 감소 | — | 4.5261% | 23.0852% |
| startup RMS | 0.01772709268 | 0.02148162406 | 0.20036294770 |
| startup 최대 절대값 | — | 0.04773694240 | 1.54342430219 |
| velocity RMS | 1.94664326116 | 1.91365466877 | 1.74055668175 |
| 직접 보호 witness | — | 실패 | 실패 |
| true 영역 margin 상한 | — | −0.19342340538 | +0.03856544940 |

startup의 원래 RMS/max 한도는 0.018/0.04다. startup 비교의 음수 margin은 solver가 이 한도와 command 목표를 동시에 만족하는 계수를 얻지 못했음을 의미한다. 표의 반환점이 startup 한도를 넘는 것을 통과로 해석하지 않는다. 근육 비교의 실제 최소 slack은 MN floor −2.3613e−12, 근육 lower −2.8910e−12, upper −1.4478e−11로 기존 허용오차 1e−12를 벗어났다. 이 미세 잔차를 clipping하거나 새 허용오차로 수용하지 않았다.

## 구현과 검산

실제 gpt-6-sol/high가 첫 구현과 첫 수정을 수행했다. 부모 FIRST 리뷰는 독립 생성 수학 180개 검사를 통과한 뒤, 실패한 초기 가능점과 계산 완료된 진단 배열의 보존을 보완하도록 요청했다. 같은 Sol이 수정했고 부모 SECOND 리뷰는 함수 21개의 AST 유지, 실제 preflight/loader를 거친 생성 실행 67개 검사, 추가 저장 배열 84개 검사를 확인했다. 총 생성 검산 331개, 집중 테스트 10개와 Ruff 검사·형식이 통과했다. 부모의 잔여 소스 수정은 필요하지 않았다.

최초 부모 생성 oracle은 QR 최대값과 직접 pointwise 최대값의 API 구분을 잘못 가정해 중단됐으며 기록을 보존했다. 부모 검산을 실제 direct API로 교정했을 뿐 구현이나 평가 기준을 바꾸지 않았다. 전체 오차를 보존하는 직교변환에서도 순간 최대값은 달라지는 생성 사례를 사용했다.

원본은 한 process에서 Clarabel solve 두 번, startup AlmostSolved/25 iteration과 muscle AlmostSolved/39 iteration으로 끝났다. 감독 3.550798700075초, 출력 4,835,444bytes이며 누적 원본 보수시간 3059.892423899961/3202초다. 자동 retry 0, 새 QR/SVD·feature·신경·물리 실행 0, F6 노출과 M2/UI 변경 0이다. CPU1 저장 행렬 계산을 실제 CUDA 보행 실행으로 집계하지 않는다.

부모 저장 검산 253개는 원래 행/열의 정확한 선택, scalar fsum QR 및 pointwise 잔차, min/max 물리 보호, true/inner support와 dual 상한을 독립 확인했다. 저장된 진단 배열에 대한 추가 38개 비교도 통과했다. 별도 gpt-6-sol/high 감사자는 작성자의 결과 해석을 읽기 전에 원시 A/problem/x/z에서 두 dual을 계산했고 약 5e−15 이내로 일치했다. 부모 저장 검산 시간 1.346179499989초는 원본 실행 비용과 별도로 기록한다.

startup 반환점에는 negative pointwise 행 12059에서 −9.8190e−10의 slack이 남고, muscle 반환점에는 envelope 행 9172에서 −5.9686e−13의 true slack이 남는다. 각 SOC 잔차도 기록했다. 배제는 반환점의 정확 가능성에 의존하지 않는 보정 dual 상한으로 판단했으며, 직접 witness는 원래 실제 계수 조건으로 별도 판단했다. 이 결과는 유한 정밀도 저장 affine 문제에 한정하고 엄밀한 구간 증명으로 부르지 않는다.

주요 증거는 `verification/banc-natural-gait-m3-20260919/`의 `e174-protection-split-freeze.json`, `parent-e174-final-source-review.json`, `e174-protection-split/result.json`, 감독 기록, `e174-parent-saved-verification/result.json`, `parent-e174-raw-capture-check.json`, `parent-e174-saved-certificate-audit.json`, `parent-e174-final-review.json`이다. 다음 방향은 `docs/BANC_M3_AFTER_E174_DIRECTION_REVIEW_20260923.md`에 기록한다.
