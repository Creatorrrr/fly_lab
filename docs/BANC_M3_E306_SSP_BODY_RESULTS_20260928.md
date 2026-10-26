# E306 실제 신체 SSP 시작 동작 결과

실제 신체를 연결한 SSP 후보는 신경 범위 오류 없이 100 control / 0.5초를 완료했지만, 원래 시작 동작 기준의 최대 관절 명령 오차를 넘었다. 실행과 원자료 검증은 유효하며, 후보 채택과 자연 보행 판정은 미달이다. 첫 신경 입력 구간은 E305와 완전히 일치했다.

| 측정 | E306 | 고정 기준 | 판정 |
| --- | ---: | ---: | --- |
| 시작 관절 명령 RMS | 0.017894977272018938 rad | ≤ 0.018 | 통과 |
| 시작 관절 명령 최대 절대 오차 | 0.06620980268835475 rad | ≤ 0.04 | 미달 |
| LTM RMS / 최대 | 0.0013637844316520794 / 0.013190561316255553 | ≤ 0.009 / 0.025 | 통과 |
| Native upright 최소 | 0.9255412735144708 | ≥ 0.9 | 통과 |
| 비발 접촉 샘플 | 0 | 0 | 통과 |

최대 오차는 42관절 배열의 0기준 index 8, control 99 / 0.495초에서 발생했다. 실제 명령 -0.0006115702967118275 rad와 참조 0.06559823239164292 rad의 차이다. 첫 최대 오차 기준 초과는 control 50 / 0.25초이며, 초과는 여러 관절에 분포한다. E150의 같은 위치 오차는 -0.03301265929973218 rad였다. E296도 같은 index 8 / control 99에서 최대 오차 0.07604739933841798 rad를 보였다. E306은 적분기와 자원 계산식이 함께 달라 이 비교만으로 개선 원인을 특정하지 않는다. E306 대조군은 첫 0.005초만 실행했으므로, 같은 SSP의 자원 감쇠 없는 0.5초 대조군이 다음 인과 구분에 필요하다.

같은 body hash의 기존 E157 actuator layout으로 index 8이 `flylab/c_thorax-lm_coxa-roll`임을 확인했다. 최대 오차는 왼쪽 가운데 다리의 몸통–coxa 회전 명령이다. 총 10개 관절에 기준 초과가 있었으며, 기록된 control 끝점에서 처음 g<1이 관측된 시점은 control 11이다. 이는 stage 최초 시점 측정이나 인과 판정이 아니다. 추가 근거 `parent-e306-saved-error-context.json`은 저장 자료만 읽은 0.04465820000041276초 / 원본 0회 분석이다.

Root는 저장된 StateStore manifest/NPY, 전체 입력 1,045개, 소스 116개, 출력 manifest 578개 및 별도 result SHA를 독립 검산했다. 원래 전체 초기 상태와 두 arm의 E305 초기/첫20tick 신경 상태는 byte exact였다. 후보의 R 최소 0.40479193004655317, 관측 끝점 g 최소 0.5738260746002197이며, 비활성 5,623 source와 원래 core로 들어가는 보호된 6,172 edge는 자원 1을 유지했다. 모든 관측 clock, 신경/물리 비용, close 반환, 종료 PID를 확인했다. 끝점 g는 stage 전체 history 측정이 아니다.

Neutral의 새 SSP와 기존 RK4 차이는 별도 수치로 보존했다. 첫 구간 rate 최대 차이 0.0006058216094970703, adaptation 0.00010120868682861328이다. 42관절 action 최대 차이는 1.0989058474852875e-6 rad다. 924개 MuJoCo 저장 state의 최대 차이 0.06483287287483108은 혼합 단위 값이며 관절 오차가 아니다. 기존 solver와 미래가 exact하다는 주장을 하지 않는다.

실제 gpt-6-sol / xhigh 구현 → root FIRST → 같은 Sol R1 → root SECOND 순서를 지켰다. FIRST에서 발견한 package 계보 충돌과 오류 publication 중 최초 예외 유실을 R1에서 고쳤으며, SECOND의 생성 검사 39개와 독립 오류 재현 검사를 통과했다. 검증용 소스의 수정은 기존 E150/M2/신경 모델/활성 UI를 변경하지 않는다.

원본 supervisor 비용은 61.50152019999223초이며 누적 8,221.508572399503 / 8,290초, 잔여 68.49142760049654초, 예약 0이다. 신체 2회, scientific neural 2,020 tick / native physical 5,050 step, setup 560 / capture 280 / settling 4,000을 구분했다. Supervisor peak RSS 3,261,812,736 bytes, 전체 출력 319,482,361 bytes. Root 원자료 검산은 passive 2.4528407999896444초, 원본 실행 0회다. 원본 비용은 이미 한 번 차감했으며 재차 차감하지 않았다.

근거는 `verification/banc-natural-gait-m3-20260919/e306-ssp-recovery-body/`, `parent-e306-original-raw-audit.json`, `parent-e306-saved-startup-localization.json`이다. Result SHA `6be280252bfa8a0a3978b0b63405f68474c5899a7615a6ad287e9c38a882e406`, output manifest SHA `836c7a7e3fa0d3da01e9e02cf726fdf3965d4a68d52a81c61adc08957d64ccb4`.

시작 기준 미달에 따라 1.5초 / 10초 / F6은 실행하지 않았다. F1–F8, F2 전체, 후보 채택, 자연 보행 및 생물학적 기전 검증 완료는 모두 false다.
