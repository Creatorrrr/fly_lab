# E156 단계 B — 같은 신경 출력의 물리 단계 전달 대조

부모 설계일 2026-09-21. 단계 A의 첫 구현/같은 Sol 첫 수정/부모 두 번째 리뷰 뒤에만 구현을 확정하고 원본 source gate를 연다. 단계 A 검증은 신체 효과의 증거가 아니다.

## 질문과 고정 대조

E155의 50개 물리 위상 전체 관측은 5ms 명령 직후 첫 1ms에 미끄럼이 집중됨을 보였다. 기존 5/10/40ms 모집 bank를 그 50개 단계에 전달하면 미끄럼을 줄이면서 전진을 보존하는지 검사한다. 새 tau, gain, 근육 기하, neural weight 또는 CPG/teacher/root 보정은 없다. 과거 M2-E22의 단일 30ms 보간 실패는 유지한다.

입력은 E150의 저장된 전체 2000 control, 정확한 391-MN rate/이전 bank/원본 96활성/이후 상태와 기존 단조 모집 파라미터다. 모든 2000 endpoint를 단계 A helper로 검산한다. 단, 실제 물리 개입은 이미 진단한 checkpoint t=1.0에서 t=2.0까지 control201..400, 1초/10000 substep 하나로 제한한다. 기준군은 E154에서 같은 checkpoint와 명령으로 qpos/qvel/ctrl 및 전체 native 표본 exact를 확인한 저장 baseline이다. baseline 물리를 다시 실행하지 않는다.

이 시험은 저장된 BANC 출력의 **신체 단독 개입**이다. 바뀐 자세가 신경 감각으로 되돌아가는 폐루프는 아직 실행하지 않는다. 후보 전진을 BANC 자율 보행 성공이라고 부르지 않는다.

## 구현 범위와 불변 조건

새 파일은 `tools/verify_c_banc_m3_substep_body.py`, `tests/test_c_banc_substep_body.py` 두 개다. 단계 A helper와 기존 body/어댑터/검증기는 읽기 전용이다. 기존 실제 Sol/high가 FIRST 구현하고 부모 리뷰에서 필요하면 같은 Sol이 첫 수정한다. 두 번째 리뷰 뒤의 잔여 수정은 부모가 맡는다.

원본 자료 및 source SHA와 source review gate를 실행 전에 검증한다. body는 E154의 `construct_body_safely`와 기존 `new_body(saved)`로 하나만 생성한다. 실제 neutral을 읽기 전용 view에 넣고, checkpoint의 adapter schema/hash/original hash/calibration/groups/parameters/roster를 보존한다. 동적 bank/원본 activation/activation만 검산할 현재 행으로 교체한다. checkpoint의 200번째 상태와 native recording의 bank/activation/original_activation이 exact여야 한다. 기록 전체 shape/dtype/finite/availability/control/network clock을 검증하고, trace 391 rate와 native rate의 대응도 대조한다.

기존 E150 parameter bundle의 tau/kappa/slopes/membership/motor_ids와 checkpoint의 대응 배열도 exact여야 한다. helper가 이전 bank와 원본활성 및 현재 rate만으로 계산하고 이미 decode된 이후 상태와 비교하게 한다. 2000행 endpoint 오차와 완료 mask를 저장하며, 실패한 행의 원본 input/이후 상태 및 예외를 먼저 남긴다. 선택된 200구간의 50×42 명령을 그대로 저장한다. 중간 명령을 새로 fit하거나 결과에 따라 선택하지 않는다.

실제 body의 첫 qpos/qvel/ctrl과 clock은 원래 trace의 tick200과 일치해야 한다. 본래 `step_joint_targets`를 control당 한 번 호출한다. 이 인스턴스의 `_step_physics`만 감싸서 각 호출 직전에 그 substep의 관절 42개 ctrl을 기록된 trajectory 행으로 대입한다. 기존 단일-tau interpolation은 비활성임을 확인한다. 다른 ctrl(6 LTM, 복부 hold, adhesion 등)은 원래 control 명령 그대로 유지하며 모든 substep에서 별도 비관절 mask로 exact 검사한다. 마지막 관절 명령과 전체 ctrl은 원래 endpoint와 exact여야 한다. 각 구간은 정확히 50step이며 추가 물리/신경 step을 호출하지 않는다. 호출 종료 시 instance method를 복원한다.

각 물리 호출 전의 적용 ctrl·tick·substep을 raw에 먼저 등록하고, 실제 적분 완료/관측 완료를 구분한다. 실제 body는 변경하지 않는 shadow mjData에서 integration state를 복사한 뒤 mj_forward 및 기존 contact_diagnostics로 모든 10000 load/slip/valid/nonfoot/upright를 기록한다. 관측 실패 때에도 이미 적용한 ctrl과 적분 계수, 부분 raw를 보존한다. 수치 오류/경고 또는 body fault는 실패로 종료한다. 예외/cleanup/저장 오류를 최초 원인과 구분하고 종료 상태·manifest를 마지막에 확정한다.

control별 실제 qpos/qvel/ctrl, body position/basis/다리 좌표와 targets를 저장한다. `capture_control`에는 실제 body와 현재 tick/rate/tendon_inputs만 전달하고 감각/네트워크를 실행하지 않는다. actor의 최종 snapshot도 별도 저장한다. UI/server/기존 checkpoint는 변경하지 않는다.

## 결과 판정 — 실행 전 고정

비교 창은 양쪽 모두 [1,2]의 anchor+200 endpoint와 (1,2]의 native10000이다. 아래는 짧은 개발 대조를 다음 폐루프 검증으로 진행할지 정하는 기준이며 F1–F8을 대체하지 않는다.

- 모든 provenance, endpoint 재구성, anchor, 비관절 ctrl, clocks와 10000 관측이 유효해야 한다. 자료 실패는 미끄럼 가설의 반증으로 세지 않는다.
- native 지원 load>=0.02BW 표본 전체의 `sum(load*slip)/sum(load)`가 동일 baseline의 80% 이하여야 한다.
- signed forward는 각 위치 차이를 이전 physical forward basis에 투영해 합한다. candidate가 baseline의 90% 이상이며 양수여야 한다. 수평 순변위/경로 비율은 baseline보다 0.05를 초과해 나빠지지 않아야 한다.
- 모든 native nonfoot=false, upright>=0.9, 수치 경고/body fault=0이어야 한다.
- 6개 기하학적 무릎의 95–5% 폭을 함께 보고하고 각 baseline의 90% 이상을 보존해야 한다. 명령 폭으로 실제 관절 폭을 대체하지 않는다.
- 전진 연속성은 동일 201경계에서 위치 미분을 물리 forward에 투영 후 Gaussian20ms를 적용해 speed>0.5mm/s 비율, 중단 수, 가장 긴 구간을 함께 보고한다. 구간 경계 smoothing을 같게 적용하며 기존 전체 10초 F4와 혼동하지 않는다.

위 gate 전부 만족할 때만 이 같은 후보를 전체 BANC 폐루프로 연결하는 다음 설계를 진행한다. 실패는 보존하고 gain/tau sweep을 시작하지 않는다. 본 대조만으로 후보 채택·생물학적 보행·최종 goal 완료를 선언하지 않는다. F6 확인 상태는 접근하지 않는다.

## 사전 자원과 기록

원본 process1/자동 재시도0, CPU thread1/GPU0/neural0/fit0. body constructor1, shadow1, constructor settling2000+scientific10000 steps, shadow forward10000. 기존 baseline 재실행0. 전체 2000 control의 helper 계산은 자료 계산이며 신경/물리 재실행으로 세지 않는다. 원본 내부65초/외부75초/감독85초, raw 최대512MiB/free disk4GiB. 기존 실제 누적2535.6120672000106초/배정2567초를 보존하고 ceil2536+85=2621초를 이번 검증의 누적 상한으로 사전 배정한다. 원본 실행 전 source gate가 필수다.

freeze, source review, supervisor, 원본 raw/result 및 부모 독립 scalar 집계는 `verification/banc-natural-gait-m3-20260919/`의 새 E156 경로에 남긴다. 검사 실패 원본을 덮어쓰거나 같은 명령을 자동 재실행하지 않는다. M2·기존 실패·F1–F8·E150 incomplete·활성 UI는 그대로 보존한다.
