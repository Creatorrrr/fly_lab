# E172: 계수 크기를 제한해도 저장 명령 근사는 개선되지만 보행 후보는 아니다

E172의 두 제한된 signed affine 가족에서 command RMS를20% 이상 줄이는 유한 witness를 얻었다. static은41.6687%, history는48.3638% 줄었다. 계수 절대값은2 이내이고 수치 해상도 지표도 사전 기준을 통과했다. 따라서 E171의 큰 계수 상쇄만으로 설명되던 표현력 문제를 이번 제한된 저장 문제에서는 해소했다.

그러나 두 witness 모두 음수 구동과 근육 예산 위반을 포함하며 startup RMS가 크게 증가했다. **실제 보행 후보로 채택하지 않는다.** 새로운 신경·물리 실행, F6 노출, UI/M2 변경은 없다. 전체 자연 보행 목표 F1–F8은 계속 미완료다.

| 항목 | 기존 baseline | static | history |
|---|---:|---:|---:|
| command RMS | 0.05668638535 | 0.03306590616 | 0.02927071717 |
| command 오차 감소 | — | 41.6687% | 48.3638% |
| 해상도 지표 RMS | 허용 0.00005668639 이하 | 7.5577e−10 | 2.6812e−9 |
| startup RMS | 0.01772709268 | 0.66745386251 | 1.72083254652 |
| velocity RMS | 1.94664326116 | 1.60554995214 | 1.26766254259 |
| 음수 alpha 개수 | — | 472/1980 | 1526/1980 |
| MN floor 위반 | — | 313/330 | 330/330 |
| 근육 하한/상한 위반 | — | 77/35 | 77/64 |
| true box projected upper | — | 0.27085918178 | 0.35454783287 |

startup의 기존 RMS 한도0.018은 변경하지 않았다. startup의 시간별 최대값, native activation, clipping, 실제 신체 보행은 이번 QR 진단으로 평가하지 않았다. history에는 음수 beta1524개와 음수 alpha+beta1516개도 있다. 음수 계수를 실제 생물학적 근육 구동으로 해석하지 않는다.

## 실행과 검토

실제 `gpt-6-sol/high`가 첫 구현과 첫 수정을 담당했다. 부모 FIRST 리뷰에서 inner dual support box, 다리별 RMS 누락, 전체 설정 기록 누락을 찾아 같은 Sol이 수정했다. 부모 SECOND 리뷰에서 R1–R3 해결을 확인했고 추가 소스 수정은 필요하지 않았다. 생성 테스트12개, Ruff 검사·형식, 독립 full-shape 수학 검산152개와 실제 preflight를 통과하는 생성 실행95개를 확인한 뒤 원본을 실행했다. 첫 생성 검산의 실패 기록과 양쪽 Sol 스냅샷을 보존했다.

원본은 단일 process에서 Clarabel solve2회로 종료했다. static24/history27 iteration, 모두 AlmostSolved 상태였으며 상태 이름 외에 직접 QR witness와 수치·입력·자원 조건을 확인했다. optimizer wall은0.203초/1.5초, 감독 전체3.278799699969초, 출력9,793,700bytes로 사전 한도 안이다. 자동 retry0, 새 QR/SVD·history recurrence·신경·물리 계산0이다. 과거 비용을 더한 원본 보수누적은3052.308218299877/3195초다.

부모의 독립 저장 검산2722개는 전역 sparse 열 배치와 직접 scalar fsum QR 잔차, 원래 sample 수의 RMS, 모든 signed endpoint, MN/근육 예산 및 witness 판정을 확인했다. 검산2.4419642초는 원본 비용과 별도로 기록했다. 별도 Sol 감사의 증명서 검산 결과는 `parent-e172-saved-certificate-audit.json`에 보존한다. 두 양수 upper는 배제 증거가 아니며 직접 witness와 모순되지 않는다. 이 결과는 유한 정밀도의 저장 affine 문제에 한정한다.

history 반환점은 수치 보호용으로 축소한 inner 계수 범위를 약1.93e−9 벗어났다. 원래 true 범위에는 최소9.81e−8의 여유가 있어, 사전에 고정한 true-box witness 조건을 충족한다. 이 수치 잔차를 숨기거나 inner 정확해로 보고하지 않는다. dual 검산과 실제 채택 불가 판정은 그대로 유지한다.

## 다음 결정에 주는 의미

이전 보호 조건을 모두 둔 E169/E170은20% 목표를 충족하지 못했고, 이번에는 계수 크기만 유지한 signed superset이 충족했다. 여러 제약을 동시에 제거했으므로 차이의 원인을 음수 허용 하나로 단정할 수 없다. 다음 비교에서는 같은 저장 신호·목표·크기에서 양수 구동 조건만 복원하여, 양수 전달 자체와 추가 근육/시작/속도 보호 조건의 영향을 구분한다. 기준을 낮추거나 이번 witness를 UI의 보행으로 연결하지 않는다.

주요 증거는 `verification/banc-natural-gait-m3-20260919/`의 `e172-bounded-qr-capacity-freeze.json`, `parent-e172-final-source-review.json`, `e172-bounded-qr-capacity/result.json`, 감독 기록, `e172-parent-saved-verification/result.json`, `parent-e172-final-review.json`이다. 후속 방향은 `docs/BANC_M3_AFTER_E172_DIRECTION_REVIEW_20260923.md`에 기록한다.
