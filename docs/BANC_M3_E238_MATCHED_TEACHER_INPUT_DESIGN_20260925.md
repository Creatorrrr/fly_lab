# E238: 같은 보행 신체에서 BANC 입력과 목표를 동시 수집

E237은 같은 반응식에 실제 E/I 입력을 넣으면 startup 실패가 재현되고, 학습 때 입력을 넣으면 조건부 예측이 재현됨을 확인했다. 따라서 다음 학습은 E150의 감각 입력과 다른 움직임의 교사 목표를 짝짓는 대신, 검증된 E85 보행 신체가 실제로 만든 입력과 그 신체에 적용한 명령을 짝짓는다. 이번 E238은 그 자료를 만드는 한 번의 수집이며 fit이나 자율 보행 성공 판정이 아니다. E03b/E16–E29의 교사 decoder·재귀학습 실패, E225 startup/주기 실패와 E236 기술 실패는 보존한다.

H238은 원래 BANC 동역학을 그대로 계산하면서 신체 명령 경계만 기록 명령으로 바꾸면, 동일한 신체 움직임과 신경 입력을 충실하게 수집할 수 있다는 것이다. 경쟁 설명은 복원 과정, 수집기 또는 명령 적용 순서가 원래 신경/물리를 바꾼다는 것이다. 두 arm을 순차로 실행하고 첫 parity가 통과한 뒤에만 교사 arm을 생성한다.

## 실행 계약

두 arm 모두 E224 `effector_canonical/start`를 `unwrap_neural_current_checkpoint`로 명시적으로 분리한 원래 `MotorTimescaleTrial`에서 시작한다. 그래프 158706개 노드와 provenance, CSR, 모든 원래 뉴런 rate/adaptation/시간상수 및 감각 encoding을 유지한다. E223 병렬 effector만 수집에 사용하지 않는다. E32는 E85의 신체 초기값 비교에만 쓰며 E32의 신경 모델을 복원하지 않는다. `SignedInputChannelNetwork`(기존 E208 클래스)로 교체한 전후 whole trial snapshot은 exact여야 한다.

`BancWalkingSession.advance`의 감각→drive→신경20tick→decode→신체50step→navigation 순서를 유지한다. 인스턴스의 `body.step_joint_targets` 경계에서 계산된 decoder 명령을 별도로 기록하고, 적용할 기록 명령으로 바꿔 원래 body 메서드를 정확히 한 번 호출한다. 일반 advance 후 신체를 다시 진행시키지 않는다. 원래 메서드/관측기/소유권은 모든 종료 경로에서 복구한다. session의 tendon_inputs는 실제 적용값으로 맞추되 신경이나 encoder 상태를 바꾸지 않는다. continuous_joint_activation=false, active_length=false, adhesion_model=legacy와 body 기본 interpolate=true를 검증한다. 모든 적용 adhesion은 false이다.

1. **parity**: E235의 control1..99 저장 command42와 실제 LTM6 및 복부2 힘줄 값을 적용한다. 99control, .495초, 1980 neural ticks, 4950 physics steps, 7920 RK stage다. E235와 100개 control의 원래 motor_rate32, body/관절/발/qpos/qvel/ctrl/applied tendon·adhesion 필드 및 4950개 native 공통 필드를 exact 비교한다. E235의 stage E/I/drive free343 subset 전부 byte exact, counters391=7920/overflow0/armed0이어야 한다. endpoint의 E235 final을 unwrap해 원래 network 전체, body 전체, underlying sensory adapter의 상태, navigation, clocks를 exact 비교한다. 사용하지 않은 canonical decoder activation·last_motor는 비교 대상이 아니다. 기록 명령 없이 원래 decoder 자체 출력을 적용하는 arm으로 대체하지 않는다.
2. **teacher**: parity를 저장·검증·해제한 뒤 같은 초기 BANC에서 E85 `commands.npz`의 2000개 관절 명령만 적용한다. 10초, 40000 neural ticks, 100000 physics steps, 160000 RK stages. adhesion=false, 8개 tendon=.2. phase/time은 학습 신경 입력으로 전달하지 않는다. E85 start의 body subtree와 동일 초기 body인지 exact 확인한다. 2001 control의 body/qpos/qvel/ctrl/관절/발 등 물리 필드와 100000 native 공통 필드를 E85와 exact 비교한다. E85의 가짜 zero motor_rate나 없는 navigation은 신경 비교에 쓰지 않는다. tendon_inputs의 control0 표시는 실제 body ctrl와 구분하고, 초기 물리 상태를 바꾸지 않는다. 최종 whole body를 E85 `final-body`와 exact 비교한다.

trace 물리 비교 필드는 body, qpos, qvel, ctrl, joint_angles_rad, joint_targets_rad, joint_tracking_error_rad, adhesion, upright, feet, fault, control_tick, time_s다. 적용 tendon 비교는 control1부터 한다. native 공통 필드는 time_s, normal_load_bw, slip_mm_s(동일 NaN 위치 포함), nonfoot_contact, upright다. parity의 slip_valid도 E235와 비교한다. 전체 신경 network와 감각 필터의 parity는 최종 checkpoint에서 확인한다. 새로운 원본 arm/재시도/조건 조정은 이번 실행 안에서 하지 않는다.

## 저장·검증·판정

각 arm의 observed-start/final whole checkpoint, 391개 원래 MN endpoint rate32, 적용 command42/tendon8/adhesion6, 별도 decoded command/tendon, control/neural/physics clocks, trace, native contact, construction/advance ledger를 저장한다. E/I/drive는 float32 `.npy`로 전체 stage와 391개 motor 순서를 보존한다. 수집기는 total/E/I/drive 네 GPU buffer를 할당하지만 호스트에는 필요한 세 채널만 저장한다. teacher의 네 GPU buffer는 1000960000B, 저장 세 채널은 750720000B다. observer는 graph capture 완료 후 한 번 arm하며 반환·stop·동기화·오류 경로를 보호한다. 실패해도 완성 control, 부분 native, stage buffer/counters, 원래 첫 오류와 추가 저장 오류를 각각 남긴다.

valid 조건은 parity 전부 통과, teacher 10초 완료, 전체 적용 기록과 E85 물리 상태 exact, 신경 rate/input finite, native nonfoot0/upright>=.9, clock·stage count·source hash·자원 한도 일치다. 통과하면 다음 유한 단계에서 이 동시 수집 자료를 사용한 학습 설계를 한다. parity가 실패하면 teacher는 not_run이며 수집 경로만 수정한다. teacher 물리가 다르면 학습하지 않고 적용 경계를 검토한다. 유효한 수집도 교사 제거 후 안정성이나 자연스러운 생물학적 보행을 증명하지 않는다. 기존 startup 기준을 E85 목표로 자동 교체하지 않는다. 후속 fit에서 목표/평가 정합을 사전에 별도로 결정하며 최종 F1–F8은 그대로 유지한다.

## 소유권·자원

실제 gpt-6-sol/high가 `tools/observe_c_banc_m3_matched_teacher_inputs.py`, `tests/test_c_banc_matched_teacher_inputs.py`만 구현한다. 부모 FIRST→필요시 같은 Sol의 첫 수정→부모 SECOND→잔여 수정은 부모 순서다. 부모는 감독기와 독립 저장검산을 작성한다. 기존 동결 파일, UI, M2, 최선 E150 및 미사용 F6 초기상태는 수정·실행하지 않는다.

원본 job1/retry0/fit0/solver0, CPU1/GPU1. 두 arm 합계 2099 control, 41980 scientific neural ticks, 104950 physics steps. 신체 생성2, 각 arm의 복원6+관측교체1=신경 생성7씩 총14를 construction profiler로 검증한다. setup interval 실행560ticks와 graph capture280ticks는 과학 시간과 분리한다. 실제 누적 setup가 예상과 다르면 진행 전에 중단한다.

내부150초(마감 저장15초 예약)/외부170초/감독180초, RSS6GiB/추가GPU4GiB/출력2GiB/최소free3GiB. E208 full10s 원본 감독76.8575초를 근거로 parity·구성·저장 여유를 둔다. 현재 원본 누적6137.263854399662/6215초와 남은77.736145600338초를 보존하고 원본 실행 전에 배정+110→6325초, 잔여187.736145600338초에서180초 예약한다. 과거 실패 비용을 초기화하지 않으며 새 전역 사용자 시간상한을 만들지 않는다. 생성 검사·읽기 전용 검산은 별도 계측한다.
