# E289–E292 재귀 전달 자원 구현과 실제 신체 결과

새 재귀 전달 자원 모델의 CPU/CUDA 계산과 whole session 연결은 구현·검증했다. **고정 후보의 자연 보행 개선은 실패했다.** E292에서 약0.3124초에 자세 유지 기준을 벗어났으므로 UI 기본값이나 채택된 보행 모델로 승격하지 않았다. 이 기능은 생물학적으로 측정한 BANC 시냅스 동역학이 아니라 공학적 가설이다.

## 구현과 검토

- `tools/banc_m3_recurrent_resource_synapses.py`: 기존 VNC premotor7136의 기존725849재귀 연결에 source별 자원을 추가했다. `dR/dt=(1−R)/tau−u*r*R`이며 r/adaptation/R를 동일 RK4 단계에서 갱신한다. CUDA는 기존 packed CSR 계산을 보존하고 새 자원 계산은float64다. u=0일 때 원래 연산 경로를 그대로 사용한다.
- `tools/banc_m3_recurrent_resource_trial.py`: 원래 E150 body/adapter/decoder와 whole state를 유지하면서 신경 동역학만 연결한다. 생성 소유권, 저장·복원, 비영 상태에서 동일 미래 진행을 검증했다.
- `tools/verify_c_banc_m3_recurrent_resource_body.py`: 원래 neutral 상태/실제 초기 안정성/조건부1.5초 보행을 순서대로 검증한다. E292에서는 .5초 전에 native health 기준이 실패했으므로 이후 단계는 실행하지 않았다.

사용자 지정대로 actual gpt-6-sol(xhigh: 신경/CUDA 구현, high: whole wrapper/신체 검증)가 개발했고 root가 검토했다. E289의 미선택 explicitzero 거부와 E290의 tuple→disk list 차이, E291의 독립실행 wall 측정 equality 문제는 각각 같은 Sol의 첫 수정 후 root SECOND에서 검증했다. 실제 실행에서 발견된 파일 publication 문제는 SECOND 이후 잔여 수정이므로 root가 고쳤다.

수정된 신경 코드는42CPUtests와3CUDAtests를 통과했다. 독립41cell/117edge7tick reference와 GPU의 최대 차이는 rate4.7683716e−7/adaptation7.4505806e−9/resource0으로 사전 허용치 안이었다. 미선택 explicit±0가 포함된 neutral captured advance도 원래 모델과 bit exact였다. Whole wrapper42tests, body runner 최종73tests 및 Ruff가 통과했다. 테스트 통과는 자연 보행 성공을 의미하지 않는다.

## 실제 실험

동일 graph/source-neural/adapter/원래 E150 neutral/decoder, u=.05/tau=.150, R초기1을 고정했다. 새 edge/CPG/시간표/교사입력/rootforce/seed탐색/decoder fit은 사용하지 않았다. Neutral 비교는 최상위 wall_s의 독립 성능 측정값만 따로 보존하며 신경·신체·control clock을 포함한 다른 과학 상태는 exact다.

| 실행 | 관측 | 판정 |
|---|---|---|
| E291 | neutral exact 통과; candidate9controls에서 ledger atomic replace WinError5 | 기술적 미완료, 보행 판정 불가 |
| E292 | neutral exact 통과; candidate62controls 완료 후63번째 interval의0.3124초 upright=.8999351733<.9, nonfoot0 | 유효한 신체 안정성 실패, 후보 기각 |

E291 실행 중 root가 PowerShell로 기록 파일을 읽었으며 Windows에서 읽기handle이 atomic replace를 막는 현상을 실제 작은 파일로 재현했다. 오류 당시 handle의 주체를 단정할 수는 없다. Root 수정은 WinError5/32/33에만 같은 기록 저장을 최대6번/총대기.155초 재시도하고 각 오류를 별도 기록한다. 과학 control은 재실행하지 않는다. E292에서는 새 저장 충돌0회였다.

E292 activity의 마지막 완성된 관측은0.31초다. 건강 검사 중단 시 신경은0.315초(1260ticks), 신체는0.3124초(3124steps)였다. Unhealthy whole final은 제공되지 않았으며 독립 network/adapter/body component checkpoint를 보존·검증했다. 이 세 시점을 동일 시점으로 취급하지 않는다.

기존모델 대비 명령 차이는 이미 .05–.10초 구간 최대.2425099rad, .20–.31초 최대2.2688762rad까지 커졌다. 이 prefix 기술통계는 완성된 .5초 startup 판정 또는 인과 증거가 아니다. 저장된 resource 최소는 .40530874였다. 자원이 변했다는 사실만으로 유용한 리듬이나 협응이 생겼다고 볼 수 없다.

![저장 원시자료의 명령 차이, 자세, 자원 변화](../verification/banc-natural-gait-m3-20260919/e292-saved-health-diagnostic-v2.png)

## 비용과 다음 결정

E291 감독48.796525700018초(신경200/물리500), E29249.285619599978초(신경1280/물리3174)를 이전 원본비용에 더했다. 각 실행의 setup560/capture280/settle4000도 별도 기록했다. 현재 원본 누적7967.464174899509/8045초, 예약0, 실행 중인 원본작업0이다. 생성 검증·수동 saved-data 검사와 실패한 첫 root partial audit도 progress checkpoint에 별도로 남겼다. 배정 확대는 실행 전에 기록했고 비용을 초기화하지 않았다.

[E293 설계](BANC_M3_E293_RESOURCE_SIGN_BALANCE_DESIGN_20260927.md)에 따라 저장된 동일 rate에서 흥분성 전달 감소와 억제성 전달 감소를 분해한다. 두 부호를 일괄 약화시킨 구조가 신경 입력의 균형을 어떻게 바꿨는지 확인한 뒤 다음 구조를 설계한다. 아직 F1–F8은 미완료이며 M2와 UI, 미사용 확인용 초기상태 F6는 보존한다.

근거는 `verification/banc-natural-gait-m3-20260919/parent-e289-e290-r1-tiny-qualification.json`, `parent-e291-original-raw-audit.json`, `parent-e292-original-raw-audit.json`, 각 original freeze/supervisor와 `e291-recurrent-resource-body/`, `e292-recurrent-resource-body/`에 있다.
