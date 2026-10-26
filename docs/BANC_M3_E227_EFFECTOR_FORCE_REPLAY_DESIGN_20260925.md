# E227: 고정 effector 명령의 신체 재현과 수평 반력 진단

## 결정과 가설

E226에서 주요 단절10개는 실제 초기 방향 후진을 포함하고 관절 추종은 양호했다. RH를 포함한 하중을 받는 발의 회수 운동이 동반됐다. 그러나 다른 후보의 E185/E186에서는 높은 slip과 실제 제동 다리가 같지 않았다. 따라서 현재 신체의 수평 반력을 먼저 확인한다. A: RH의 지지 중 회수 운동이 주요 음의 반력과 일치한다. B: 다른 다리 또는 다리 간 상쇄가 주로 관여한다. C: 정확한 재현·관측이 되지 않아 판별할 수 없다. 어느 다리도 사전에 원인으로 확정하지 않는다.

현재 후보의 기록 명령을 신체 모델에서만 재생한다. 기존 E71/E75에서 저장 명령 재생의 실제 물리/native 동일성을 검증한 방법을 재사용하되, 이번 후보에서도 직접 일치를 확인해야 한다. 전체 BANC를 재실행한 결과나 새로운 자율보행 후보로 주장하지 않는다. 학습·계수변경·명령 retiming·보간·root/발 위치 강제·관측에 맞춘 상태 교정은 없다.

## 실행과 유효성

1. E224 effector의3초 `final/`에서 신체만 `tools.analyze_c_banc_observation.new_body`로 구성·복원한다. GraphStore/신경망/CUDA는 생성하지 않는다. 원래 seed/world/parameters/body를 쓰고 즉시 신체 snapshot, model hash와 control600 물리값을 비교한다.
2. 원본 E225의 control601..1663, 정확히1063개 명령을 원래5ms/50 native steps로 적용한다. 관절42, 실제 adhesion, 모든 tendon(복부 포함)을 그대로 쓰고 원래 `interpolate=True` 경로를 사용한다. E225의3.0001..8.315초53150 native 표본까지 총 추가5.315초 물리만 진행한다.
3. 매 control의 qpos/qvel/ctrl, body 위치/basis/다리 geometry, 실제·목표각/오차, tendon/adhesion, upright/feet/fault가 원본과 정확히 같아야 한다. 기록 motor rate는 비교용 metadata이며 계산하지 않는다. neural_request/navigation 등 신경·항법 상태를 재현했다고 주장하지 않는다. native의 시간/normal load/slip/valid/nonfoot/upright 전 표본은 dtype/shape/bytes(의도적 NaN 포함)로 동일해야 한다. 불일치는 기술 실패이며 힘 해석을 중단한다.
4. 기존 E185의 `install_native_observer`와 `shadow_force_sample`을 재사용해 같은 shadow `mj_forward`에서 힘을 얻는다. 원래 물리 상태에는 손대지 않는다. 여섯 다리의 모든 마디 힘, 전체 fly 바닥 힘, 잔차, 수평 heading, thorax/COM 위치를 최소 원자료로 보존한다. segment 수준 임시 배열은 기존 함수가 계산하되 저장은 필요한 고정 필드로 한정한다. 질량/BW/좌표축·힘의 작용 방향·body inventory를 기록한다.
5. 건강 guard는 기존100us nonfinite/nonfoot/upright<.9/body fault 그대로다. 부분 실패도 실제 clock 및 native 진행량, raw, 오류, 비용을 보존하며 자동 재시도하지 않는다. 건강하고 완전히 일치한 control에서만 신체 snapshot을 저장한다. 시작/최종 및 E226 주요10사건 entry-80ms에서 snapshot을 남겨 후속 개입의 정확한 신체 시작을 확보한다. 이것은 신경 상태를 포함한 whole checkpoint가 아니다.
6. E226의 주요10사건과 고정 pre[-80,-20), during[entry,exit), post[exit+20,exit+80)ms를 그대로 쓴다. 각 제어행 i의 native 구간은 `(i-1)*5ms < t <= i*5ms`다. native 단계에서 수평 force의 양/음 부호를 먼저 나누고, 하중≥.02BW 조건별 시간합·지지 시간·평균과 다리별 비중을 기록한다. 무힘·무지지 분모0은 unavailable이다. 모든 지정 사건·모든 다리를 보존한다. 음의 힘 크기 순위는 제동 반력의 관측이며 개별 근육의 인과 판정이 아니다.

## 단위와 판정의 한계

힘은 native z-up MuJoCo 좌표의 post-integration shadow 순간값이다. 초기/현재 수평 heading을 수직 성분0으로 정규화해 투영한다. body-axis force에는 pitch의 수직 지지 성분이 포함되므로 수평 추진으로 혼용하지 않는다. BW는 기존 전체 fly 질량×중력이다. 시간합은 shadow 반력의 수치 합이며 직전 solver impulse나 실제 근육 힘이 아니다.

유효성은 동일한 물리 재현과 완전한 관측으로 판단한다. RH 또는 다른 다리의 음의 힘이 큰 것은 다음 제한된 개입의 근거이며 자연 보행 성공 점수가 아니다. 새로운 통계적 유의성/생물학적 반복/최종 F3–F5 판정은 만들지 않는다. E225의 resource_limited와 10초 미완료를 유지한다.

## 구현과 단계 자원

실제 gpt-6-sol/high가 `tools/observe_c_banc_m3_effector_force_replay.py`와 `tests/test_c_banc_effector_force_replay.py`를 구현한다. 부모 FIRST→필요시 같은Sol 첫수정→부모SECOND→필요시 부모 잔여수정. 작은 생성/mock 검사는 sample 정렬·힘 부호/좌표·비지원 nullable·불일치/부분 진행·snapshot 범위를 확인한다. 부모는 원시 힘의 합/투영/사건 mask를 독립 검산한다.

이전 E71/E75의 신체만 재현 비용은 약4–6벽초/물리초였고, 이번에는 전체 다리 힘을 추가 관측한다. 신체 구성·5.315초 재현·force 관측·저장/정리를 포함해 CPU1 / GPU0 / 프로세스1 / retry0 / 내부65초(마감8초 예약) / 외부75초 / 감독80초, RSS6GiB / 출력128MiB / free2GiB를 사전 한도로 둔다. neural/solver/fit0, 과학 물리 최대53150steps다. 신체 초기 settling은 구성 비용과 함께 별도 기록하며 직접 계측과 소스 유래 수를 구분한다.

사용자 지정 전체 토큰·시간 한도는 없다. 이번 새 물리 진단을 위해 **단계 원본 자료 실험 배정에80초를 명시적으로 추가하여6004→6084초**로 잇는다. E226 회계 정정을 포함한 누적5990.496209299676초는 보존되어 총 잔여93.503790700324316초 중80초를 예약한다. E225의 실패한95초 예약은 변경하지 않고 원본 재시도로 처리하지 않는다. E227은 새 지식 공백을 다루는 신체 전용 관측이며, 근거 없이 기존 실패 후보를 반복하는 시간이 아니다. 부모 검산·순수 생성 검사 비용도 별도로 기록한다.

F1–F8, E150 최선 검증 산출물, M2/UI, F6 미노출을 보존한다. 새로운 과학 기준·최종 목표·유저 승인을 만들어 내지 않는다. 원래 개발·검증 권한 안에서 수행하며 새로운 외부 지출·배포·운영 변경은 없다.
