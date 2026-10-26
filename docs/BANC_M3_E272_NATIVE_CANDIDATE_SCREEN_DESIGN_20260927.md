# E272 — 같은 source 계수 후보의 실제 재귀망·신체 판정

2026-09-27 root 사전 설계. E270의 결과를 보기 전에 후속 판정을 명시한다. 이 문서는 실행 예약이 아니며 E269/E270/E271의 root 검토와 E270 조건부 지지 뒤에만 원본 실행 계약을 동결한다. F1–F8, 모든 과거 실패, M2/E150, 현재 UI와 미노출 F6를 보존한다.

## 동일 후보와 기준

E270의 최종 한 번 float32로 반올림한1660×3 계수만 사용한다. 전체 BANC 노드/CSR/부호/원래막τ·gain·adaptation·E150 전달·감각·신체·tendon·physics step을 고정한다. stage 기록이나 teacher 명령은 실제 신경망에 공급하지 않는다. source 필터는 실제 각 RK4 단계의 source rate로5/40ms 이력을 갱신한다. 양성 결과를 만들기 위한 외부 CPG/phase/root/발 접촉 교정은 없다.

기준은 E150 candidate-neutral의 동일 전체 checkpoint와, E257 baseline이 E150의 처음301 controls/15000 native samples/301 recruitment rows를 exact 재현한 저장 자료다. baseline 0.5≤t<1.5초 전진1.6790085779712631mm, slip과 나머지 수치는 고정한 arm-result/raw에서 읽어 같은 평가 함수로 재확인한다. 저장 전진 값을 다른 시간 구간의 결과와 비교하지 않는다.

## 단계적 실제 시험

1. E271 wrapper에 neutral 계수를 넣어 .005초 한 control만 실행한다. 최초 전체 checkpoint에서 허용된 source layer/schema/patch 외 항목이 exact이고, 원래 E150 첫 control 및50개 native 접촉 샘플과 같아야 한다. 원본 관찰기의 별도 MjData 경로를 사용하며 생성 중 settle과 GPU warmup/capture를 별도로 기록한다. 이것이 실패하면 후보 신체 실행0이다.
2. 새로 복원한 동일 neutral 전체 상태에서 E270 한 후보를 시작한다. native100µs observer는 원래 upright≥.9,nonfoot0,finite/fault 조건을 매 step 검사한다. control1..99 startup은 E103 원래42q와6LTM 목표, qRMS≤.018/max≤.04,LTMRMS≤.009/max≤.025로 평가한다. .5초에서 startup이 실패하면 raw와 실제 신경/물리 clock을 보존하고 중단한다.
3. startup이 통과하면 같은 세션을1.5초까지 연속 실행한다. E257 `short_health`, `short_window_slip`, `short_forward`, 기존 biological gait analyzer를 그대로 사용한다. baseline보다 전진이 나쁘지 않고, supported slip이 크지 않고, longest eligible forward bout≥.3초이며 baseline보다 짧지 않아야 한다. 완전한 보행 주기 수도 baseline보다 증가해야 한다(현재 baseline 여섯 다리 모두0). 이 짧은 단계의 주기 증가는 여섯 다리10주기/반주기 협응이라는 최종 F4를 대신하지 않는다.
4. 동일 후보가 위 조건을 모두 통과할 때에만 .5초 제외10초 F3–F5, whole checkpoint disk future/motor cut, 미사용 초기상태 F6 및 기존 목표/stop/restart F7과 영상/F8로 진행한다. 해당 실행들은 재학습 없이 같은 계수/전달을 유지하고 각각 원본 한도와 source freeze를 사전에 기록한다.

## 증거·실패·자원

신경/신체 실제 생성·setup·과학 tick/physics step·control·CPU/GPU wall/RSS/VRAM을 구분한다. trace/control command, 실제391MN+51source rate/adaptation/drive와 source history, muscle bank/activation, native 접촉/슬립, 초기·마지막 전체 checkpoint, finite journal·완료범위·오류·파일 SHA를 보존한다. 장애/건강중단/원래startup미달/짧은보행미달을 서로 구분한다. 관측되지 않은 값은 pass로 간주하지 않는다.

현재 원본 예약0이며 E270 조건부 gate와 E271 구현 검토를 통과하기 전에는 어떠한 body도 생성하지 않는다. 실제 조건부 개선은 재귀망/신체 개선의 증거가 아니고, 실제 공학적 보행 개선도 생물학적 기전 검증을 대신하지 않는다.
