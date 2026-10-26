# E226: 저장 자료로 effector 보행의 주기별 전진 단절을 분해

## 질문과 판별 범위

E225는 resource_limited 부분 결과다. 동일 0.5–8.315초에서 E150보다 순전진과 전진 구간 지속성이 개선됐지만 완전 적격 주기는 여전히 0이다. 독립 Sol/medium의 저장 자료 검토에서는 10개 적격 구간이 0.37–0.40초이고 한 구간 안에 연속 발 전방 극대점 둘이 없었다. 일반적인 발 관측 누락보다는 주기별 짧은 전진 단절이 직접 병목이다. 원래 기준이나 이벤트 prominence를 바꾸지 않는다.

다음 질문은 단절 시점에 (A) 실제 병진 속도가 감소하는지, (B) 방향 회전 때문에 몸체 전방 투영만 감소하는지, (C) 어느 발의 지지·미끄러짐·관절 추종 및 근육 명령 변화가 함께 나타나는지다. 저장된 수직 하중과 발 움직임만으로 수평 지면 반력이나 인과 기여를 단정하지 않는다. 후보 교정에 필요한 다음 개입의 범위를 좁히는 진단이며 이번 단계에서 파라미터를 바꾸거나 채택하지 않는다.

## 자료와 고정 분석

- E224 effector control/native prefix와 E225 continuation control/native suffix, 부모가 정확한 연결을 확인한 `parent-e225-partial-trace.json`, E224 effector `start/`의 anatomy/calibration을 사용한다. 값·dtype·시간축·중복 연결 행과 trace 명령 일치를 확인한다. 1664행 / native83150표본, 마지막8.315초다. 입력/source hash를 실행 전에 동결한다.
- 기존 `analyze_c_banc_biological_gait.load_model`의 위치 미분·20ms Gaussian·전방0.5mm/s 기준을 그대로 사용한다. `[0.5,8.315)`에서 moving 구간 사이에 양쪽이 관측된 below-threshold 구간을 모두 기록한다. 개별 sample 시간뿐 아니라 sample-count 기반 duration도 저장한다. 전방3초 이후이며 직전 moving 구간이0.3초 이상인 단절을 주요 비교 집합으로 고정한다. 전체 단절도 숨기지 않는다.
- 단절 전 60ms(`entry-80ms`부터 `entry-20ms`), 단절 전체, 회복 뒤60ms(`exit+20ms`부터 `exit+80ms`)를 비교한다. 끝에서 자료가 부족하면 해당 통계는 unavailable로 남기고 다른 구간으로 대체하지 않는다. 이벤트 중심 원자료의 ±100ms 정렬 창도 저장한다.
- control 시각마다 물리 world 위치·발말단 위치·42개 실제 관절 및 명령을 사용한다. 원래 전방 속도, 초기 방향 투영 속도, 수평 world 속도, yaw rate, 몸체 상대 발 전후 위치/속도, 발 world 수평 속도/전방 투영, 관절 명령/실제 각도/오차·미분을 계산한다. 미분과 smoothing은 진단 전용이다.
- native 100us 표본50개를 실제 `(previous_control_time,current_control_time]`에 정확히 묶는다. 평균/최대 하중, support≥기록된0.02BW, 지지 발 수, 하중 가중 valid slip을 계산한다. 무하중/invalid slip을0으로 대체하지 않고 분모0이면 unavailable로 보존한다. 처음control에는 직전 interval이 없으므로 native 집계를 available로 만들지 않는다.
- 96근육 activation과 소유 leg/muscle metadata를 기록한다. 명령 변화는 고정 `0.02 * moment @ activation`의 근육별 기여로 정확히 분해한다. 전/단절/후 delta의 크기 순위는 기술 통계이며 힘·인과 효과의 순위가 아니다. 새 무릎·계수·시점 최적화는 없다.
- 모든 주요 이벤트별 원값/평균, 전체 집합 요약, 예외 및 관측 한계를 저장한다. 기존 gait summary는 그대로 재사용하며 새로운 성공 점수나 임계값을 만들지 않는다.

## 구현, 검증, 자원

실제 gpt-6-sol/high가 `tools/analyze_c_banc_m3_effector_breaks.py`와 `tests/test_c_banc_effector_breaks.py`만 구현한다. 부모 FIRST 후 필요시 같은 Sol의 첫 수정, 부모 SECOND 후 잔여 오류는 부모가 처리한다. 부모는 저장 결과를 독립 계산으로 확인하고 개입 선택을 설계한다.

순수 생성 검사에서는 native/control 경계, 중복·누락 거부, 지지 없는 slip의 nullable 처리, 이벤트 창의 오른쪽 끝 불완전성, 실제 병진 감소와 회전 투영의 분리, canonical 기여 합을 확인한다. 실제 자료 분석은 source 승인 뒤 CPU1 프로세스1 / retry0 / 내부20초·외부25초·감독30초 / RSS6GiB / 출력128MiB 이내다. 저장 자료 분석만 하므로 원본 dynamics/CUDA/physics/solver/fit/model construction/새 데이터 수집은 모두0이다. root 및 agent의 분석·리뷰 비용은 별도로 보존한다.

현재 원본 누적5986.0027711996845 / 6004초, 잔여17.997228800315497초는 그대로다. 과거 시간 제한을 완화하거나 원본 재시도를 이 단계에 숨기지 않는다. 관측 상관이 명확하지 않으면 불확실로 남기며, 유망한 기전이 있어도 원래 신경 입력·해부학 제약을 지킨 새 개입과 실제 폐루프 확인을 별도 사전 설계한다. F1–F8 미완료, F6 미노출, E150 최선/M2/UI 보존.
