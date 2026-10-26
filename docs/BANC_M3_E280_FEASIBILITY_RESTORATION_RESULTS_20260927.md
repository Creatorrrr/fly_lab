# E280: 제약 보정 방법의 여섯 다리 조건부 검증

고정된 여섯 거절 제안을 CUDA 조건부 신경 엔진에서 재현한 뒤, 새 Jacobian을 사용하는 최소 norm 보정을 최대 두 번 적용했다. **6/6 모두 원래 시작 제약을 만족하고 기존에 채택했던 작은 이동보다 학습 목적값이 낮아졌다.** 과거 before/trial 출력·목적값과 before gradient는 모두 오차 0으로 재현됐다. Root가 저장 원자료의 목적값, 엄격 제약, Armijo, primal/dual 인증을 별도로 계산해 같은 판정을 확인했다.

| 다리 | 보정 횟수 | 기존 비교 목적값 | 보정 후 목적값 | 비교점 대비 감소 |
| --- | ---: | ---: | ---: | ---: |
| LF | 1 | .985762 | .971407 | 1.456% |
| LM | 2 | .871213 | .806194 | 7.463% |
| LH | 2 | .918739 | .842019 | 8.351% |
| RF | 1 | .966320 | .939213 | 2.805% |
| RM | 1 | 1.000000 | .994926 | .507% |
| RH | 1 | .945565 | .895764 | 5.267% |

이는 첫 학습 단계의 **목적값** 비교다. 완성 모델의 RMS 개선율, 실제 보행 품질, 전체 망의 동작을 의미하지 않는다. 다리별 fixed SSE는 원래 E277 값을 사용했으며 여섯 보정 결과를 합친 후보를 만들지 않았다. LM/LH의 첫 보정은 실제 제약 미달로 거절했고 두 번째 보정에서 통과했다. 실제 판정에 solver 잔차 허용오차를 사용하지 않았다.

구현·첫 수정은 gpt-6-sol/xhigh, FIRST/SECOND 검토와 잔여 수정은 root가 수행했다. 생성 검사 59개와 Ruff가 통과했다. 원본은 조건부 엔진 12개(max 2 live), Clarabel 8회/137 iterations, 계수 평가 20개, forward/derivative 각각 520,000 ticks다. 전체 BANC·신체·물리·새 full fit·F6는 실행하지 않았다. Supervisor 10.498171799990814초, peak RSS 1,456,873,472 bytes, 출력 153,177,549 bytes이며 소유 프로세스는 모두 종료됐다. 원본 누적 7833.773759199601/7905초, 잔여 71.22624080039895초다. 생성 검사와 수동 검산 비용은 별도 기록했다.

다음 설계에 보정 단계를 적용할 근거는 확보했다. E277의 train 20%·LM 개발 기준 미달과 E279의 실제 시작 max .0408695 > .04 기각은 유지한다. 특히 중립 초기 출력도 max 약 .03946/RMS .017727이므로, 초기 상태를 고려하지 않고 학습 제약을 .036/.0162로 일괄 축소하면 중립점부터 불가능해진다. 저장 Jacobian과 초기 전이를 검토해 제약 여유를 설계한 뒤 새 학습을 동결한다. 아직 후보를 채택하지 않았으며 F1–F8과 자연스러운 생물학적 보행은 미완료다.

근거: [사전 설계](BANC_M3_E280_NONLINEAR_FEASIBILITY_RESTORATION_DESIGN_20260927.md), [root 독립 검산](../verification/banc-natural-gait-m3-20260919/parent-e280-original-final-review.json), [실행 비용](../verification/banc-natural-gait-m3-20260919/parent-e280-original-supervisor.json), [출력 해시](../verification/banc-natural-gait-m3-20260919/parent-e280-output-manifest.json).


2026-09-27 후속 정정: 위 max 2 live는 당시 meter의 논리적 context 수다. 다음 단계 검토에서 evaluate closure가 다음 엔진 생성까지 살아 있는 경로를 발견했고, 생성 weakref 검사에서 최대 4개 holder가 겹쳤다. E280 실제 엔진 생존 한도 준수 보장은 철회한다. 실제 GPU peak는 미계측이다. 독립 검산한 여섯 수치 결과와 원본 비용은 유지하며, 이 수치 근거만 다음 방법 설계에 사용한다. [자원 근거 정정](../verification/banc-natural-gait-m3-20260919/parent-e280-resource-qualification.json)을 함께 읽어야 한다.
