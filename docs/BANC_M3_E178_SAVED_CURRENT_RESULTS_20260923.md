# E178 결과: 운동뉴런 억제의 출처를 확인했지만 보행 후보는 미채택

E177 두 팔의0/1/1.1/10초 전체 신경 상태로7개 세포의 입력56행을 재구성했다. 실제 CUDA 경계 전류와 모두 bit exact였고, 부모가 원본 행17872개 연결 기여를 독립적으로1040회 검산했다. CSR는 graph topology, navigation 보정, 기존 calibration130943개와 individual fit1572개, 후보의5개 feedback 변경으로 정확히 재구성된다. 부호·단위 변환 또는 저장 전류 구현의 불일치는 이 범위에서 발견되지 않았다.

## 확인한 입력

출력이 없던 두 RM tibia-flexor MN은 동일한 `IN13A001`(index144453, root720575941645048993)의 억제 입력을 크게 받았다. 이 source의 BANC annotation은 verified GABA이며, 아래 efficacy는 측정된 생리학 값이 아니라 기존 individual fit의 모델 값이다. 두 MN 자체의 predicted GABA는 verified annotation이 없으며 NMJ 생물학을 입증하지 않는다.

|10초 경계|MN108052 baseline|MN108052 candidate|MN153939 baseline|MN153939 candidate|
|---|---:|---:|---:|---:|
|전체 recurrent current|−985.2385|−1041.1228|−609.9285|−645.0947|
|양의 입력 합|11.1818|11.9527|0.8181|1.0304|
|IN13A001 기여|−984.9970|−1040.4909|−610.5513|−644.9493|
|MN rate|0|0|0|0|

전류 숫자는 rate-model 단위이며 mV/근력으로 해석하지 않는다. IN13A001→MN 실제 weight는 각각−14.63609695/−9.07219887이며 기본 contact×.03 값의 약4.8787/4.8775배다. 1/1.1/10초의 두 팔 전체에서 이 source가 차지한 음의 입력 비중은 MN108052에서95.19–98.95%, MN153939에서99.58–99.97%였다. 해당 행의 navigation 전류는0이었으므로 이번 관측은 목표 제어 억제 입력의 과대증폭 가설을 지지하지 않는다.

이 dominant 항을 뺀 나머지 경계 입력에도 threshold7.5를 적용하면 유효 입력은 각각−10.68~−7.74, −7.77~−6.88이다. 이는 저장 합의 항목별 분해이며 연결 제거 후 폐루프 예측이 아니다. 단일 억제 연결을 줄이는 것만으로 발화나 자연 보행이 회복된다고 결론 내리지 않는다. 전체 tibia-flexor pool의 다른 MN과 근육 전달을 확인해야 한다.

감각 source30981/107685도10초 경계에서 recurrent current가 음수였다. baseline에서는 각각−23.8861/−18.0904, candidate에서는−49.0804/−69.6219다. 이 둘은 기존446cell parameter cohort에 없어 threshold/cap을 임의로 채우지 않았다. 감각 구동량을 올리는 다음 실험도 아직 승인된 후보로 간주하지 않는다.

## 수치·구현 검증

실제gpt-6-sol/high 최초구현→부모FIRST→같은Sol첫수정→부모SECOND를 수행했다. 부모 생성 scalar47검사, 완전 생성 IO7검사와 관련 pytest58개가 통과했다. FIRST에서 FTZ 누락, outer control tick 검증, manifest 실패 시 success 잔류, candidate parameter 검증을 고쳤다. 초기 부모 fixture의 graph manifest 누락도 실패 자료와 함께 보존했다.

설치 CuPy는 CSR에 FTZ를 적용한다. 두 팔 모두 packed이며 unique weight bits2971개다.37개 edge 기록에서 FTZ 차이가 있어 이를 명시적으로 재현했다. 원래 bit exact 기준은 유지했고 실제 신경 커널은 변경하지 않았다. 원본 분석은 검토된 source hash로 한 번만 실행했다. 이후 부모가 Ruff format 미통과를 정리했으며 두 파일 AST가 실행본과 동일함을 검증했다. Ruff check와 format check는 최종 통과다. 원래 실행 source는 e178-sol-first-correction에 보존하며 기존 source gate를 재사용하지 않는다.

분석 process1/retry0, supervisor2.5314770999830216초, 부모 raw 검산0.5827411999925971초 별도다. 신경·물리·GPU·모델 생성·fit은0이다. 원본 보수 누적은3420.470999800018/3656초다. 원본 입력71개/재사용 source92개/보호 증거3296개의 보존을 별도 검산한다.

후보 미채택, F6 노출0, M2/UI 변경0, F1–F8 및 전체 목표 active를 유지한다. E177의 F3 통과/F4·F5 실패와 최대 연속 전진.215초는 바뀌지 않았다. 본 결과는 입력 출처 진단이며 자연스러운 보행 검증이 아니다.

주요 증거: verification/banc-natural-gait-m3-20260919/e178-saved-current/, parent-e178-raw-review.json, parent-e178-source-review.json, parent-e178-format-review.json, parent-e178-final-review.json.
