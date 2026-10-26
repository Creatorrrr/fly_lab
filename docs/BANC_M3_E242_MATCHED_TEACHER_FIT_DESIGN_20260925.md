# E242: 같은 입력의 관절 명령과 속도에 대한 단일 양수 계수 fit

E241의 입력 기저와 원래 decoder 재구성이 유효했다. H242는 실제 건강한 E240 보행의 같은 시점 q/속도를 앞6초에서 학습한 근육 공유 반응이 뒤4초에서도 원래 decoder보다 정확할 수 있다는 것이다. 원래 신경 입력·해부학·생리·신체와 이전 실패를 유지한다. 이 비교의 성공은 조건부 후보를 물리 검사에 넘길 근거이며 자율 보행 성공은 아니다.

## 고정 자료와 계수

E241 parameters/prototypes/B/H/D_group/q_intercept/canonical-original/targets/groups와 부모 검산을 동결한다. scale/중심을 다시 만들지 않는다. 변수는 원래 free90근육×64=5760개 beta다. 같은 근육의 원래 MN만 계수를 공유하고,각 MN은 자신의 입력/기억/중심으로 반응한다. beta 범위는[0,1], LTM6행은0이고 제외48MN 출력은 매시점의 실제 E240 rate다. 초기 activation·held q0·.02 moment·10ms EMA와 전체391MN/96근육/42관절을 유지한다. 기존 E218 beta는 새 중심에 이식하지 않는다.

## 목적함수와 실제 최적화

각다리7관절의 학습 control1..1200을 사용한다. 속도는 control c와c−1의 차이/.005이며 c=1..1200이다. 검증 control1201..2000 및 그 속도는 목적함수·제약·정규화 척도에 쓰지 않는다. 다리별 양의 정규화 척도 s_q/s_v는 E241 원래 decoder의 **학습** q/속도 RMS다. 실제 숫자는 freeze에 기록하고 원본 자료에서 재계산하여 일치해야 한다. 0/비유한 척도는 오류로 처리한다.

다리마다 J=.02 moment·D_group을 그대로 구성하고, 정규화한 q residual과 속도 residual을 행 방향으로 붙인다. 목적은 두 정규화 제곱합과 기존 작은 ridge(각다리1e−8/그다리변수수·sum(beta²))의 합이다. 전체 QP는6다리의 block diagonal P/선형항과 단일 box 제약 A=[−I;I], b=[0;1]이다. 원래 E218 group_anatomy/group_joint_matrix와 quadratic_objective의 수학은 재사용할 수 있다. E197 startup 제약, E150 phase-remapped target loader, E218 고정19cone builder는 호출하지 않는다. Startup도 새 교사와 같은 시간에 맞추는 학습 자료이며 과거 .018/.04 proxy로 바꾸지 않는다.

Clarabel solver 생성/solve 각1회, max_threads1/max_iter150/time_limit90초, gap abs/rel 및 feasibility tolerance1e−8을 사용한다. solve 전 전체90초와 저장10초 예약이 내부150초에 들어가야 한다. solver raw x/s/z와 info·P/선형항/A/b/컬럼·상수·정규화·설정은 판정 전에 저장한다. 단일 solve 뒤 coefficient 재탐색·다른 창 fit·validation을 본 재학습은 하지 않는다. solver의 시간 한도는 soft임을 기록하고 외부 감독으로 상한을 지킨다.

## 재구성과 후보 검사

먼저 Solved 상태, 유한 raw, 실제 box·primal/dual residual과 objective를 검산한다. 원래 raw beta가[0,1]에서 벗어나는 최대량이1e−8보다 크면 후보를 만들지 않는다. 그 이하의 수치 반올림만[0,1]에 투영할 수 있으며 raw/투영량/투영 전후 목적과 오차를 모두 남긴다. 이 투영을 원래 해를 exact라고 부르지 않는다. 후보의 전체96×64 beta에서 LTM6행은 정확히0이어야 한다.

원래 B/H에서 얻은391개 rate64와, free effector를 float32로 변환하고 제외48개 원래 rate32를 유지한 실제 decoder 입력을 모두 재구성한다. q/activation/tendon의 float64 및 float32 경로와 affine q, 초기 held0를 저장하고 비교한다. 실제 native 연결 전 단계이므로 이 자료의 재생을 새로운 신경 실행으로 세지 않는다. tendon은 E240와 같아야 하고 오차한도는2e−12다. 두 경로의 차이는 보고하되 후보 검사는 실제 float32 decoder 경로를 기준으로 한다.

후속 **신체 검사에 넘길 조건부 후보**의 사전 기준은 다음과 같다. 학습과 뒤4초 각각에서 전체 q RMS가 원래 decoder의80%이하이고 전체 속도 RMS는 원래보다 나쁘지 않아야 한다. 각다리 q/속도 RMS도 두 구간 모두 원래보다 나쁘지 않아야 한다. 새 교사의 startup1..99 q RMS는 원래의80%이하, 최대오차는 원래이하여야 한다. 이는 초기부터 오차를 줄인 후보를 선별하는 기준이지 물리적으로 충분한 정확도라는 주장이 아니다. 통과 여부와 무관하게 다리별·전체·startup 오차와 최대값을 원단위로 제공한다.

유효한 후보가 위 기준을 만족하면 다음에 그 명령의 동일 신체 재생을 사전 설계하고 원래 F3–F5를 검사한다. 이를 통과한 뒤 교사 없는 전체 BANC 폐루프, 저장·차단·확인·기능회귀를 수행한다. 후보가 시간 분리 검증이나 startup에서 실패하면 그 수치를 유지하고 실패 구간/입력 표현 또는 물리 전달의 판별 근거로 다음 방법을 정한다. 자동 재fit나 기준 완화는 없다. F1–F8/F6미노출/기존M2·E150·UI는 변하지 않는다.

## 개발·자원

실제 gpt-6-sol/high가 새 `tools/fit_c_banc_m3_matched_teacher.py`, `tests/test_c_banc_matched_teacher_fit.py`만 구현한다. 기존 solver 수학을 새 학습 정렬·box·float32 readout·회계와 연결하는 작업이라 high를 선택한다. 부모 FIRST→같은 Sol 첫수정→부모 SECOND→잔여는부모 순서다. 생성 CPU 검사는 작은 행렬의 q/속도 목적·ridge·gradient·box와 원래컬럼 조립, 검증자료 미사용, held/fixed/float32 및 실패 raw/receipt를 확인한다. 새 수학 kernel과 solver 구현 변경이 없으므로 생성 CUDA/생성 solver0이다. 작은 수식 검산과 기존 solver 근거를 재사용한다.

원본 저장 fit1회/retry0/solver1, CPU1/GPU0/전체BANC0/신체0/physics0다. E218 기존 감독57.84초에 근거하여 내부150초(저장10초 예약)/solver90초/외부170초/감독180초, RSS6GiB/출력384MiB/free3GiB를 고정한다. 이전 누적6574.145712599628/6680초·잔여105.85428740037241초를 보존하고 실행전배정+75→6755초·잔여180.85428740037241초에서180초를 예약한다. 생성 검사·부모 저장 검산은 별도 계측하며 사용자 전역 예산은 새로 만들지 않는다. 실패 결과와 실제 호출·자원·부분 raw를 보존하고 실패 후 success/완료 상태를 남기지 않는다.
