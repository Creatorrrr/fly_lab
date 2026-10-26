# E153 같은 명령 재생의 다리별 접촉력 진단

E152에서 E150의 전진 중단은 미끄럼 증가와 LH/RM/RH 지지 중 앞방향 발 운동에 동반됐다. 이는 발 원점의 운동학적 연관성이며 제동력이나 근육 원인을 증명하지 않는다. 다음 질문은 **같은 운동 명령에서 어느 발의 실제 지면 접촉력이 추진 또는 제동 방향인가**다. 새로운 제어기, fitting, 신경 실행, F6, UI 또는 후보 채택은 없다. F1–F8은 불변이다.

## 실행 및 동일성

E150 `checkpoint-1s`를 기존 `analyze_c_banc_observation.new_body`로 신체만 복원한다. 부모의 사전 schema 확인은 StateStore 읽기 1회, 모델·물리0, .061562초였다. 기존 생성자의 2000 settling physics step(.2초)을 별도 계수하고 복원 이후 scientific replay와 합치지 않는다. 몸체 모델1개, shadow mjData1개, BANC/GraphStore/CUDA/fit0이다. 생성·복원·cleanup도 시간 한도에 포함한다.

trace anchor tick200(t=1s)의 qpos/qvel/ctrl, body hash, physics time을 검증한다. 원래 trace ticks201..400의 200개 joint_targets/adhesion/tendon_inputs 및 neural_request external_force_at_start를 순서 그대로 사용한다. 각 control5ms, physics50*100us, 총1초/10000physics다. cuts.motor에 따른 interpolate를 원래처럼 적용한다. snapshot parameter에는 graded_adhesion/active_length/continuous activation 변경이 없음을 읽어서 확인하며 지원하지 않는 새 신체 설정은 조용히 무시하지 않는다.

복원 후 매 control qpos/qvel/ctrl은 원본과 `array_equal`이어야 한다. 매 physics는 기존 E150과 같은 shadow integration copy→mj_forward 관찰이며 actor에 관찰용 forward·pose 보정·state 덮어쓰기를 하지 않는다. native time/load/slip/slip_valid/nonfoot/upright는 원래 native[10000:20000]과 정확히 일치해야 한다(NaN 위치도 동일). time 예상값 비교만 atol2e-8이고 원본 time 비교는 exact다. checkpoint replay가 다르면 힘 결과로 E150을 설명하지 않는다.

## 힘 규약 및 raw 자료

MuJoCo 공식 computation contact 문서는 force가 geom1에서 geom2로 작용하며 geom1 Jacobian의 부호가 음수임을 명시한다. API의 mj_contactForce는 contact frame에서 force:torque 6차원을 반환한다. frame의 행은 world 좌표의 접촉 축이다. 따라서 foot이 geom2면 +1, geom1이면 -1을 써 `F_world = sign * frame.T @ wrench[:3] / weight0`를 구한다. feet map의 segment//5가 다리 인덱스(LF/LM/LH/RF/RM/RH)다. 제외된 contact와 foot-ground가 아닌 접촉은 집계하지 않는다. adhesion 반력이 포함되며 조직 힘 또는 전체 추진 원인과 동일하지 않다.

출처: https://mujoco.readthedocs.io/en/stable/computation/index.html#contact 및 https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce . 생성자료로 geom 순서/축 회전/부호를 검증하고 부모는 간단한 sphere-plane MuJoCo forward-only 모델에서 위 방향 및 마찰 방향을 독립 검산한다(원래 FlyBody0/적분0).

각 shadow 시점의 body rotation[:,0]에서 native z 성분을0으로 두고 정규화한 수평 앞축 h를 사용한다. `F_forward = dot(F_world,h)`는 양수 추진/음수 제동이다. 몸 기울기의 수직 하중이 전후 추진력에 섞이지 않게 수평 투영을 한다. 전 다리 vector 합도 저장한다. torque는 본 질문에 필요하지 않으므로 새 평가를 만들지 않는다.

native-forces.npz: 10000개의 time_s, control_tick(201..400 각각50), rotation_world(3x3), heading_horizontal(3), force_world_bw(6x3), force_forward_bw(6), normal_load_bw(6), slip_mm_s(6), slip_valid(6), nonfoot_contact, upright. contacts.npz: offsets 길이10001, contact_id, geom1, geom2, segment, leg, sign, frame(3x3), wrench(6), position_world(3). 각 native 샘플의 ground-foot nonexcluded contact를 원래 순서로 모두 저장한다. context.json에 feet map, support geom IDs, weight0, body hash, 실제 physics dt, source slice 및 축 규약을 기록한다. replay-controls.npz에 201행 anchor+미래의 tick/time/qpos/qvel/ctrl과 verified mask를 저장한다. 원시 관측은 assertion/집계 전에 메모리에 추가하고 finally에서 오류 전 기록까지 저장한다. 원본 실패 시 raw 일부도 보존한다.

## 사전 고정 집계

E152의 검증된 E150 features.npz에서 endpoint speed를 가져온다. 원래 tick201..400의 `speed>.5`와 나머지 두 mask를 각각 해당 50native samples에 동일하게 대응시킨다. 새 smoothing, lag 탐색, event 선택은 없다. 이 구간은 exploratory diagnostic이며 formal eligible bout/F4/F5를 재정의하지 않는다.

전체/forward/interrupted에 대해 sample 수, 다리별 및 전체 평균 signed forward force, 양의 impulse와 음의 impulse(BW*s, sum(max/min(F,0))*.0001), supported(load>=.02)에서 negative force fraction, mean normal load, supported normal-load-weighted slip(sum(load*slip)/sum(load))를 저장한다. total impulse는 먼저 여섯 힘을 합친 후 양/음 분리하며 다리별 양/음 합과 구분한다. empty mask/denominator0은 null이다. 원래 force values는 smoothing하지 않는다. 측정된 braking은 접촉 수준의 사실일 뿐 원래 MN/근육/감각 중 특정 원인의 causal isolation이 아니다.

## 개발·검토·자원

부모 설계→실제 Sol/high FIRST 구현(`tools/analyze_c_banc_m3_signed_contact.py`, `tests/test_c_banc_signed_contact.py`)→부모 FIRST review→같은 Sol FIRST correction→부모 SECOND review→잔여는 부모 직접 수정. 기존 고정명세의 body replay/수치 집계이며 새 neural/CUDA architecture가 아니므로 기존 실제 high를 재사용한다. 테스트는 생성자료만, Sol은 원래 body/model/실험을 실행하지 않는다.

CLI --freeze --freeze-sha256 --source-review --source-review-sha256 --output. 원본 읽기/모델 생성 전에 exact freeze SHA, `accepts_single_body_diagnostic=true`, int original_processes_allowed=1, 두 owned source SHA 및 입력/재사용소스/보존 SHA를 검증한다. output은 새 빈 디렉터리. 전후 SHA를 확인한다. accepted=false provisional result를 먼저 저장하고 완료 후만 true로 바꾼다. 최종 publication 실패나 한도 초과는 false로 남긴다. ledger/progress/timing/error/output manifest에 실제 read/model/settle/replay/forward 수와 마지막 저장 시간을 포함한다. 새 original 프로세스1/재시도0, CPU1/GPU0, 내부35초/외부40초/감독48초, raw512MiB/free12GiB. 기존 누적2465.3204491초를 보존하며 ceil2466+48=2514초로 기존 총한도 내에서 사전 배정한다. 기존 E150 incomplete는 유지한다.

부모 독립 검산은 raw contact별 scalar frame 변환/geom 분류/합·수평 투영·집계·원본 qpos/qvel/ctrl/native를 재검사한다. 부모 saved review는 CPU1/내부30/외부40초, original model/neural/physics0으로 별도 보고한다. 이 진단만으로 새로운 gain 탐색이나 원래 재시도는 허용하지 않으며, 이후 개입은 근거와 자원을 다시 설계한다. 실패·관측 한계도 진행 기록에 남긴다.
