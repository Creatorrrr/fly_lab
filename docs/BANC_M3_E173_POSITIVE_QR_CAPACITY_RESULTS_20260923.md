# E173: 양수 static 명령 근사는 개선됐지만 전체 비교는 미완료

E173 원본 실행은 history solver의 `InsufficientProgress`로 종료했으며 전체 판정은 **invalid_not_tested**다. 완료된 static 부분은 사전 설계의 보존 규칙에 따라 별도로 검산했다. static에서는 모든 계수가 [0,2] 범위에 있는 상태로 저장 command RMS를 25.2228% 줄였다. 따라서 적어도 이 static 저장 문제에서는 양수성 하나만으로 20% 개선이 불가능하다는 설명이 성립하지 않는다.

그러나 같은 계수의 startup RMS는 0.465258로 기존 한도 0.018을 넘고, 근육 출력 보호 조건도 위반한다. **보행 후보로 채택하지 않는다.** history의 원시 반환값은 보존했지만 인증하거나 재시도하지 않았다. 전체 BANC의 자연 보행 F1–F8은 여전히 미완료다.

| static 항목 | baseline | E173 반환점 |
|---|---:|---:|
| command RMS | 0.05668638535 | 0.04238851751 |
| command 오차 감소 | — | 25.2227545% |
| velocity RMS | 1.94664326116 | 1.68623901271 |
| startup RMS | 0.01772709268 | 0.46525806543 |
| 계수 최소/최대 | — | 9.9915e−8 / 1.99999989959 |
| MN floor 위반 | — | 36/330 |
| 근육 하한/상한 위반 | — | 10/77, 71/77 |
| command 해상도 지표 RMS | 허용 0.00005668639 이하 | 5.9309e−10 |

모든 다리의 command 및 velocity norm은 baseline보다 작다. startup의 시간별 최대값, native 활성 재구성, clipping 및 실제 신체 보행은 평가하지 않았다. 전역 25% 개선을 모든 다리의 동일 비율 개선으로 해석하지 않는다.

## 구현과 검토

실제 `gpt-6-sol/high`가 첫 구현과 첫 수정을 수행했다. 부모 FIRST 리뷰에서 반환점의 inner/true 선형 slack 기록과 포맷 검사 누락을 지적했고 같은 Sol이 수정했다. 부모 SECOND 리뷰는 기존 수학 함수 22개의 AST가 유지됐음을 확인하고, 생성 수학/전체 경로/slack 검산 총 365개와 테스트 18개, Ruff 검사·형식을 확인했다. 부모의 잔여 소스 수정은 필요하지 않았다.

원본 실행은 단일 process, solve 2회, retry 0회다. static은 AlmostSolved/25 iteration, history는 InsufficientProgress/41 iteration이었다. history 상태는 사전 허용 상태가 아니므로 exit 1과 최초 오류를 유지했다. x/s/z가 유한하다는 사실로 history의 허용 상태나 witness를 대체하지 않았다. 전체 실험의 source gate는 소비됐으며 기존 경로를 다시 실행하지 않는다.

원본 감독 시간 4.033406900009초, 출력 9,442,692bytes, 원본 보수누적 3056.341625199886/3198초다. 새 QR/SVD·history 상태 생성·신경·물리 실행은 0이다. 별도 부모 저장 검산은 1422개, 0.661377700046초로 기록한다. F6 노출과 M2/UI 변경도 없다.

## static 부분 증거의 범위

부모는 저장된 static QR의 각 행을 scalar fsum으로 재평가하고 전역 sparse 열 배치, RMS 분모, 계수 범위, 실제 min/max에 따른 MN/근육 보호, 해상도 지표와 witness를 검산했다. 별도 `gpt-6-sol/high` 감사자는 작성자 결과를 읽기 전에 static A/M/q/offset/baseline/SOC를 E172와 bit-exact 비교하고 dual 및 반환 slack을 계산했다. 두 검산은 일치했다.

true-box projected upper는 +0.06528461160373923이고 직접 반환 t는 +0.06528443175968136이다. 양수 upper 자체는 배제 증거가 아니며, 여기서는 직접 양수 witness가 개선 가능성을 뒷받침한다. 수치용 inner의 계수 하한 행 1943에는 −8.5044e−11 slack이 남지만 원래 true 범위의 최소 slack은 +9.9915e−8이고 SOC gap은 +5.2580e−13이다. inner 정확해로 부르거나 잔차를 숨기지 않는다. 이는 유한 정밀도 저장 affine 문제에 대한 검산이며 엄밀한 구간 증명이나 실제 보행 검증은 아니다.

주요 증거는 `verification/banc-natural-gait-m3-20260919/`의 `e173-positive-qr-capacity-freeze.json`, `parent-e173-final-source-review.json`, `e173-positive-qr-capacity/result.json`, 감독 기록, `parent-e173-original-failure-review.json`, `e173-parent-static-saved-verification/result.json`, `parent-e173-static-certificate-audit.json`, `parent-e173-final-review.json`이다. 후속 방향은 `docs/BANC_M3_AFTER_E173_DIRECTION_REVIEW_20260923.md`에 기록한다.
