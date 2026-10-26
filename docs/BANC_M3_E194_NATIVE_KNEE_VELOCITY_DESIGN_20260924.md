# E194: 저장 native 상태로 무릎 속도 표집 점검

2026-09-24 원본 실행 전 설계. E193의 전체 감각 관측과 독립 검산은 progress다. 속도 감각이 약한 이유로5ms 경계에서 이미 감속한 순간만 읽는 설명이 남았다. 원본 누적4758.634181599718/4994초, 잔여235.365818400282초와 F1–F8, M2/UI, F6미노출을 유지한다.

## 기존 증거를 재사용하는 결정

E193에서 evaluation hook137개 중132개, club314개 중313개의 rate가 정확히0이었다. 각도 차분과 경계 속도의 큰 차이는 사후 관측이며 원인 확정이 아니다. 과거 별도 보행 기록에는 몸통의 유사한 경계 표집 문제가 있었지만 이번 무릎의 증명은 아니다. M2 E22와 E150 E156의 연속 목표각/중간 명령 적용, M2 E59/60의2.5ms 제어는 이미 기각됐으므로 이번에는 그 개입을 반복하지 않는다.

읽기 감사에서 E157 `native-replay.npz`에 동일 E150 t=1–2초의100µs qpos/qvel/ctrl 10000행이 저장돼 있음을 확인했다. 헤더는 qpos(10000,109), qvel(10000,108), ctrl(10000,82) float64 및 시계/완료 표지이며, 결과 metadata는 accepted/publication_complete이다. E157 body hash는 E193과 같다. 따라서 새 보행이나 새10000단계 적분 없이 저장 상태에서 기하 관측을 재계산한다. 힘/접촉의 신뢰는 기존 E157/E154의 수용된 기록에 의존하며 새 힘 검증이라고 부르지 않는다.

## 구현과 입력

새 `tools/analyze_c_banc_m3_native_knee_velocity.py`, `tests/test_c_banc_native_knee_velocity.py`를 실제 gpt-6-sol/high가 구현한다. 부모 설계→Sol FIRST→부모 FIRST→같은Sol 첫 수정→부모 SECOND/잔여 수정. 모델 재구성과 파생 상태의 시점 정렬 때문에 high를 사용한다. 부모가 metadata/freeze/source review, 감독, 독립 검산을 담당한다. 기존 도구·신체·인코더·자료는 수정하지 않는다.

입력은 hash로 고정한 E157 native-replay/layout/result 및 수용된 부모 review, E150 checkpoint-1s의 전체 파일, E193 encoder/provenance/3초 baseline trace 및 final review다. E193 native observer와 E157의 전처리·원래 body factory의 소스도 고정한다. 원본 전에 소스/입력/설계 SHA 및 부모 승인이 필요하며 기존 출력은 덮어쓰지 않는다. 이 작업에 full BANC graph나 CUDA 계산은 필요 없다.

실행 전 E157 native_index=0..9999, control_tick=201..400의50개씩 반복, t=1.0001..2.0, 모든 integration/state/observation/source_verified 표지 및 finite qpos/qvel/ctrl를 검증한다. native 각 control의 마지막 행은 E193 trace201..400의 qpos/qvel/ctrl와 exact여야 한다. checkpoint는 control200/t1.0, E193 trace200과 exact여야 한다. 불일치를 새 허용치나 다른 입력으로 회피하지 않는다.

## 재구성 및 비교

1. 기존 `analyze_c_banc_observation.new_body(saved)`를 `analyze_c_banc_m3_signed_contact.construct_body_safely`로 호출해 정확한 body modelHash를 복원하고 생성 중2000 settling step을 별도 계수한다. 신경망은 만들지 않는다. 이후 actor의 `_step_physics` 호출을 금지한다. 실제 MuJoCo timestep/integrator/servo hash·시작 snapshot을 기록한다.
2. 별도 MjData의 anchor는 완전한 checkpoint integration state로 설정한다. 이후 저장 qpos/qvel/ctrl와 같은 시각을 설정하고 `mj_forward`한 뒤 `shadow.knee_kinematics(include_direction=False)`로 여섯 무릎의 각도와 해석적 속도를 읽는다. actor의 qpos/qvel/integration state/시계는 읽기 전후 exact 불변이어야 한다. qpos/qvel만으로 기하를 계산한 것이며 원래 힘·활성 상태를 모두 복원했다고 주장하지 않는다.
3. t1.0 anchor를 포함한10001행의 time, flexion6, analytic_velocity6, hinge_qvel6, source_index 및 complete/error를 저장한다.10000개 저장 상태를 빠짐없이 읽는다. 각50번째 경계(201개)의 기하 각/속도는 E193 encoder tick200..400과 atol1e-12/rtol0로 일치해야 한다. 원래 qpos/qvel/ctrl는 tolerance 없이 비교한다. 누락/예외 시 앞서 수집한 부분 자료와 원래 예외를 보존하고 재실행하지 않는다.
4. 고정200구간에 대해 실제 각도 변화, native 해석 속도의 사다리꼴 적분과 오른쪽 직사각형 적분, 양끝 경계 속도의 사다리꼴 적분을 함께 저장한다. 실제 시간 간격을 사용하며 구간은50 native steps다. 전체 구간과 각 다리의 변화량 RMS, 두 적분 오차 RMS, native/경계 속도 RMS와 peak를 모두 보고한다. 유한 차분을 순간속도의 정답으로 바꾸지 않는다.
5. 구간별 각도변화의 RMS가0이면 상대오차는null·정지로 표시한다. native 사다리꼴 상대오차≤0.1이고 경계 적분 오차가 native 오차의4배 이상이면서 상대오차≥0.25인 다리가5개 이상이면 이 저장 궤적에서 경계 표집이 무릎 운동 정보를 잃는 설명을 지지한다. 그렇지 않으면 미결 또는 해당 설명이 약함으로 보고하며 다른 원인을 단정하지 않는다. 오른쪽 적분은 integrator 차이를 보존하는 별도 진단이고 판정식을 사후 바꾸는 수단이 아니다.

이 결과로 신경 입력을 자동 변경하지 않는다. 지지가 확인되면 같은 수용기 식과 gain을 유지하면서 실제 과거 주기 내 관측을 인코더에 전달하는 인과적 수집 방식의 후속 대조를 설계한다. 외부 CPG·미래 상태·목표 궤적·gain sweep을 도입하지 않는다.

## 제한과 종료

원본1process/CPU·BLAS1/GPU0/새신경0/새과학적 적분0(생성시 settling2000 별도)/새모델1/기하readout10001 이하/자동retry0. 내부50초, 외부55초+소유 자식cleanup5초=감독60초를 같은 잔여에서 예약한다. 입력기존자료 보존/출력256MiB/free1GiB. 실제 감독 wall을4758.634181599718에 가산하며 준비·생성시험·부모검산은 별도, 미집계는0이 아니다. 다른 프로세스 종료·UI 조작은 없다.

시험은 고정50단계 정렬, 알고 있는 smooth/transient 각도·속도 적분 예, 잘못된 control/shape/누락 플래그/소스 해시 거부, 관측 오류 뒤 부분행 보존·actor 불변·cleanup 복원을 검증한다. 부모는 저장 결과에서 별도 식으로200구간 적분/비율/판정과 E193 경계 일치를 검산한다. 새 모델 적분은 검산에 추가하지 않는다. 아직 E194 원본0, 후보미채택, 전체 자연보행/F1–F8은active·미완료다.
