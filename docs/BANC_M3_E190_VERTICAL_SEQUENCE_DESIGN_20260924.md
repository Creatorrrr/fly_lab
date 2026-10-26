# E190: 저장 접촉 자료의 수직 명령·운동·제동 순서

2026-09-24. 부모 설계. 기존 finite-goal/F1–F8과 원본 누적4654.9093643997185/4994초를 승계한다. E189 low의 신경→근육→반력 효과는 확인됐지만 지속 보행은 개선되지 않았고 high는 자원 부적격이다. 같은 gain sweep을 반복하지 않는다. 실제 gpt-6-sol/high의 읽기 감사에서 E161 수직 운동과 E166 전진 표본 반례를 함께 확인했다.

## 결정에 필요한 질문

기존 E161은 RM 지지·제동에서 femur/tarsus의 아래쪽 기여를 평균했지만, E166 전진 표본에도 같은 부호가 있었다. 하강 자체를 정지 원인으로 단정할 수 없다. 이번에는 같은 고정 정지 사건 안에서 수직 요청·실제 관절 기여가 음의 AP 힘보다 먼저 변하는지, 양의 AP 힘 표본과도 같은 관계인지 확인한다. 경쟁 설명은 위치 서보와 접촉 기하/하중 변화이며, 관측의 선후도 인과 증명이 아니다.

이는 저장 자료 진단 1회다. 신규 신경/물리/CUDA/모델/fit/후보/holdout 노출은0이다. 현재 MN 입력이 실제 위치까지 전달되는 E189 결과를 보존하고, E162 group70 고정 감쇠·E177 감각5edge·E181–183 시간응답·E169/E175 저장 fit의 실패를 새로운 시도로 재포장하지 않는다.

## 고정 입력과 공통 구간

E157 native-replay.npz, E161 raw-derived.npz, E185 ground-reaction.npz와 baseline/native-contact.npz, E186 aligned.npz, E157 layout.json 및 각 원래 부모 검산을 해시로 고정한다. 구현은 NPZ 공개 header만 확인할 수 있으며 원본 값은 승인된 단일 실행에서만 읽는다. 초기 header helper의 비공개 NumPy 함수 부재와 trace 구조 탐색 오류는 준비 오류이고 원본 분석 실행이 아니었다.

E157의10000 native 표본/control tick201..400을 E185의 native index10000..19999와 직접 대조한다. 원래 저장 native time, rotation/heading, 6다리 world force, normal load, slip와 유효 mask/nonfoot/upright가 exact여야 한다. force는 E157 force_world_bw 대 E185 leg_force_bw를 대조하며 pitch 포함 forward와 planar heading을 혼동하지 않는다. E161 native_load/slip/actual RM qvel을 E157과 exact 대조한다. E161의 실제 기하학적 수직 분해를 새 신체로 재생하지 않는다. 사용한 공동 관측의 일치만 인증하며 E185에 없는 모든 native qpos/qvel의 직접 대조를 주장하지 않는다.

E161의 trace RM targets 전체 중 control tick200..400을 E186 joint_target_rad_42[:,28:35]와 exact 비교한다. control tick과 실제 time은 별도로 확인한다. 어느 대조든 불일치하면 원자료 비교는 invalid/not-tested로 종료하며 join tolerance나 구간을 바꾸지 않는다.

E185/E189의 고정8개 post 시작 control [131,219,308,391,433,480,521,564] 중 앞10/뒤10control이 공통 구간에 완전히 있는219,308,391만 사용한다. event start c의 pre tick c−10..c−1, post c..c+9이고, 각 control tick에는 원래50개 native 표본이 속한다. 전후 각각10control/50ms, 총1000 native/event, 겹침과 반복 사건은 독립 생물학 표본이 아니다. 나머지5사건은 공통창 부재로 명시적으로 제외하며 다시 검출하지 않는다.

## 계산과 출력

RM index4, femur-pitch local joint4, tarsus-pitch local joint6이며 layout 이름으로 검증한다. command_z는 E161 native_rm_command_projection_z의15개 근육(65..79) 합이고 LTM78은 별도 tendon 경로라 이 위치 moment 합에 없음을 명시한다. 실제 z는 native_z_total/root/own7[:,4,4]/own7[:,4,6]/own7 합과 residual을 보존한다. 이 값은 접촉점 기하 투영이며 개별 근육의 실제 힘이 아니다.

모든20개 5ms bin에서 (a) 전체 native AP signed/positive/negative magnitude와 normal-load 평균, (b) support≥.02BW이면서 contact_available인 표본의 하중 가중 command_z, total/root/femur/tarsus/own7 z, support 수/하중 분모를 저장한다. positive와 negative AP 표본은 같은 고정 bin에서 부호만 나누며0은 별도 제외 수다. 각 조건의 표본수·분모·가용성·하강 비율을 함께 기록하고 빈 조건은null/NaN+valid=false로 남긴다. NaN을0으로 바꾸거나 양/음 힘을 먼저 평균한 뒤 부호를 분류하지 않는다.

사건별20bin 시계열에서 command_z와 femur+tarsus z 각각에 대해 braking magnitude 평균과의 lag −2,−1,0,+1,+2 control(−10..+10ms) Pearson 상관을 모두 저장한다. x(t)와 braking(t+lag)를 비교하므로 양의 lag는 x가 먼저임을 뜻한다. 해당 사건 내부에서만 두 신호가 모두 가용한 표본을 쓰며 최소8쌍·각 분산>0일 때만 값이 있다. 최고 lag를 고르거나 유리한 사건을 pool해 인과 방향으로 선언하지 않는다. 경계 밖/무접촉/상수·저표본은 가용성 false다.

원본 native 배열의 필요한 수직/명령/AP/load/support/time/tick/유효 mask와 모든 event bin·조건·lag 수치, 제외 사건, exact 대조 결과를 NPZ/JSON에 보존한다. 고정창 반례와 양의 힘에서도 같은 하강이 관측되는 경우를 함께 보고한다. 음의 힘보다 앞선 하강이 반복돼도 제한된 상대시점 가설의 후속 개입 근거일 뿐이고, 반대·혼합·부족하면 방향 미결이다. 어떤 결과도 후보를 자동 생성/채택하지 않는다.

## 구현·자원·리뷰

실제 gpt-6-sol/medium이 tools/analyze_c_banc_m3_vertical_sequence.py와 tests/test_c_banc_vertical_sequence.py 두 파일만 구현한다. 고정 배열 조인·하중 가중·시차 계산이며 새로운 신경 수식이 없어 medium이다. 부모 FIRST→같은 Sol 첫 수정(필요 시)→부모 SECOND→부모 잔여 수정 순서를 유지한다. 합성 검사는 off-by-one/정해진 선행 신호/무접촉·0분산·부호별 분모/원본 불일치 거절에 집중한다.

source review는 부모가 실제 두 소스 해시와 frozen 입력·설계를 결합하며, CLI는 이를 원본 값 읽기 전에 검증한다. 원본 실행은 CPU affinity1/BLAS1/GPU0, 내부25초/외부35초/감독40초, 출력32MiB/free1GiB, process1/재시도0으로 고정한다. 입력 source 해시 사후 대조와 저장·정리 비용을 포함한다. 기존 잔여339.0906356002815초 안에서40초를 예약하며 실패도 누적한다. 준비·테스트·부모 독립 검산은 별도이며 미집계는0이 아니다. 소스/입력 원본은 수정하지 않는다. 기존 evaluator, M2/UI/F6와 goal active를 유지한다.
