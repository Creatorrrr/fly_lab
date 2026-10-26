# E155 실제 RM 접촉점 속도 분해

E150의 원래 1–2초 중 200개 제어 경계에서 실제 RM 접촉 134개를 재구성했다. 원래 하중·미끄럼·몸통 회전과 일치했고, 부모의 독립 scalar 재계산 및 보존 파일 검사가 통과했다. **이 결과는 운동학적 분해이며 보행 개선이나 특정 관절의 인과 효과를 입증하지 않는다.**

## 검증과 개발 절차

부모 설계 → 실제 Sol/high 최초 구현 → 부모 첫 리뷰 → 같은 Sol/high 첫 수정 → 부모 두 번째 리뷰를 완료했다. 시간 누적값을 완전 일치로 요구하던 오류, 최종 progress 기록 때문에 manifest가 틀리는 오류, 실패한 Jacobian 원시값이 빠지는 오류를 수정했다. 추가 부모 production 수정은 필요하지 않았다.

22개 테스트와 393개 독립 생성 계산 검사를 통과했다. 작은 별도 MuJoCo chain의 Jacobian은 위치의 수치 미분과 최대 3.44e-11에서 일치했다. 실제 출력 검산은 원래 자료·파일 해시·배열·집계를 합한 14,358개 검사이며, 이를 독립 실험 횟수로 세지 않는다. 모든 원본 파일을 보존했다.

단일 원본 프로세스 exit0, timeout 없음. 신체 모델1개와 shadow1개를 만들었고, 생성자 settling2000 step과 기하 계산 forward200회를 사용했다. 새 scientific physics·신경·CUDA·fit은0이다. 감독 포함 23.612067초, 보수 누적2535.612067초로 사전2567초 이내다. 원본 실행 재시도는0이다.

## 원래 200개 경계의 고정 집계

아래 값은 실제 접촉 하중으로 가중한 전방 상대속도(mm/s)다. 양수는 지면에 대해 발이 몸통 전방으로 움직이는 성분이다. root는 자유관절6개, RM은 해당 다리의7개 구동관절, 나머지는 다른 DOF의 합이다.

| 원래 endpoint 집합 | 표본 / 지지 | root | RM 7관절 합 | 나머지 | 전체 발 접촉점 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전진 | 145 / 96 | +0.017773 | +0.026598 | −0.000054 | +0.044317 |
| 중단 | 55 / 37 | +0.209839 | −0.169356 | +0.000377 | +0.040860 |
| RM 지지·제동 | 56 / 56 | +0.305441 | −0.286600 | −0.000235 | +0.018606 |
| RM 지지·추진 | 77 / 77 | −0.118607 | +0.181104 | +0.000279 | +0.062776 |

중단 집합의 RM 관절 기여는 coxa roll −0.105614, trochanter roll −0.104178, tibia pitch +0.067032mm/s 등이 상쇄한다. RM의 구동관절 전체는 이 시점의 root 성분을 상당 부분 상쇄한다. 따라서 기존 제동력 관측만으로 RM 출력 전체를 낮추거나 특정 관절을 잘못된 방향이라고 단정할 근거는 부족하다.

![RM 접촉점 속도 분해](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/e155-parent-figures-v2/rm-contact-velocity-decomposition.png)

첫 그림의 긴 축 이름이 겹쳐, 같은 수치에 짧은 이름을 적용한 v2를 표시한다. 첫 그림과 양쪽 provenance도 보존했다.

## 후속 saved-data 관측과 한계

사후 관측에서 RM의 endpoint 하중 가중 미끄럼은 전진0.11063, 중단0.07691mm/s였다. 이것은 실제 발 원점이 아닌 접촉점의 값이다. 5ms 경계만으로 전체 native 미끄럼을 설명할 수 없다.

이미 저장된50개 native phase를 모두 유지해 다시 집계하니 전체 다리의1–2초 미끄럼은1.84149mm/s인데, 각5ms 구간의 끝만 보면0.14414mm/s였다. 명령 적용 후 첫1ms는 전체 시간의20%이며 하중 합의18.812%를 차지하지만, 하중×미끄럼 합의54.606%가 이때 발생했다. 이 시간적 연관은 개입 효과를 증명하지 않는다.

현재 신체 경로는 최종 관절 명령을 다음5ms 동안 고정한다. 다음 후보 검토는 특정 RM 관절의 감쇠보다 **기존 근육 동역학을 물리 substep에 전달하는 방법**에 초점을 둔다. 과거 M2-E22의 단일30ms 지수 보간은 미끄럼을 악화시켜 기각됐으므로 그대로 다시 채택하지 않는다. 현재 E150의5/10/40ms 단조 모집 모델에 맞춘 인과적 중간 상태를 사용하고, 실제 신체 대조로 개선과 회귀를 확인해야 한다.

이 단계에서 후보는 바뀌지 않았다. E150 전체 incomplete, F4/F5 실패, F6 미노출, 기존 M2와 활성 UI 상태를 유지한다. 같은 최종 후보의 F1–F8은 미완료이며 goal은 active다.

## 근거

- [원본 결과](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/e155-rm-contact-jacobian/result.json)
- [부모 독립 검산](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e155-output-review.json)
- [사후 접촉 방향 해석](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e155-saved-interpretation.json)
- [50개 물리 phase 관측](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e155-control-phase-observation.json)
- [최종 source review](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e155-final-source-review.json)
