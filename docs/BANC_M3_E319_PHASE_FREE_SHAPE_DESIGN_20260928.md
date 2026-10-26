# E319: 절대 위상을 고정하지 않는 다리 내 운동 형태 목적함수

E107의 고정 shape-only 외부 입력 실험은 E105/E106보다 긴 전진 bout와 적격 주기를 보였다. 이것은 BANC 자율 성공은 아니지만, 이미 반복하는 E150 신경을 더 빠르게 흔드는 것보다 주기 내 관절 운동 형태를 바꿀 근거다. E319는 이 근거를 조건부 신경 학습의 목적함수로 표현한다. loss 감소는 보행 개선과 구분하고 기존 F1–F8을 그대로 유지한다.

학습의 자유 변수는 후속 계약에서 정할 현재 전체 BANC의 기존 P→P 연결 효능이며, 고정 decoder의 anatomical support와 실제 감각 입력, SSP 동역학을 유지한다. 이 구현 단계에서 효능을 업데이트하거나 원본을 읽지 않는다. E317/E318 검토와 별개로 진행 가능한 순수 생성 수식이다.

## 목적함수와 인터페이스

신규 `tools/banc_m3_phase_free_shape.py`, `tests/test_c_banc_phase_free_shape.py` 두 파일만 actual gpt-6-sol/high가 구현한다. 단순한 닫힌 곡선의 최근접 선분 수식이므로 high를 사용한다. Root 설계 → Sol → root FIRST → 필요한 같은 Sol R1 → root SECOND → root 잔여 수정을 따른다. 기존 source는 변경하지 않는다.

API는 `PhaseFreeShapeObjective(curves, startup_q, startup_tendon, *, device='cpu')`와 `.evaluate(q, tendon)`다. 곡선은 float64 (6,M,7), M은3..512이며 마지막 vertex와 첫 vertex를 연결하는 주기 곡선이다. 실제 E85 곡선 생성은 후속 원본 계약에서 한 번 고정하고 수행하며, 여기서는 생성 배열만 쓴다. Reference 계수는 소유 복사하고 학습하지 않는다. Startup q/tendon reference는 원래 control1..99의 float64 (99,42)/(99,6)이다. Evaluate는 control1부터 연속된 q(T,42), tendon(T,6), T101..320, 동일 장치 float64를 받는다. 입력을 변경하지 않고 모든 수치는 유한해야 한다.

후속 학습의 길이는0–1.5초·300control로 계획하되 이 순수 함수는 위 범위 T를 처리한다. 시작 보호는 정확히control1..99이며 shape 평가 구간은control101..T다. 이는 첫0.5초가 끝난 뒤 실제 적용되는 명령이다. 시간에 따라 teacher의 절대 phase/frequency를 목표로 넣지 않는다.

다리 l의 각7관절 명령 x를 닫힌 곡선의 모든 선분 a→b에 투영한다. `d=b-a`, `u=clip(dot(x-a,d)/dot(d,d),0,1)`, `p=a+u*d`, `D=sum((x-p)^2)/7`. 영 길이 선분은 p=a로 처리해 division NaN을 피한다. 다리마다 최소 D를 사용하고 시간/여섯 다리 평균을 scalar `shape_loss`로 반환한다. 결과는 radians squared이며 joint별 임의 재가중이나 수치 기준 보정은 없다. 기준 곡선의 vertex 시작 index를 순환 이동해도 같은 결과다. 반대 방향 traversal도 같은 곡선 거리이므로 **운동 방향·주파수·협응·전진을 loss 하나가 보장하지 않는다**. 이를 감추지 않고 native startup/short/10초 및 F4/F5 실제 판정으로 걸러낸다.

결과 dict는 `shape_loss`, `per_leg_shape_loss`(6,), `startup_q_rms`, `startup_q_max`, `startup_tendon_rms`, `startup_tendon_max`의 differentiable tensor와 `startup_pass` boolean scalar tensor다. Startup limits는 기존 .018/.04rad, .009/.025와 동일하다. Candidate 값을 보고 바꾸지 않는다. Max의 동률/최근접 선분 동률에서는 PyTorch가 택한 유효 subgradient를 사용하며 매끈한 지점의 미분 정확성과 구분한다. Hot evaluate는 CUDA device predicate를 쓰고 CPU scalar extraction/copy나 장치 동기화가 없어야 한다.

## 검증과 후속

생성 seed31901, CPU1/host1GiB, Sol 전체60초/root60초. 원본 값/모델/신경/신체/fit/CUDA 모두0. Scalar NumPy 독립 선분 투영과 forward≤1e-12, 매끈한 생성점의 q directional finite difference 상대≤1e-6 또는 절대≤1e-9, 닫힌 마지막 선분·영 길이·순환 index 불변·q/tendon 초기 경계99/100/101·각 기존 startup gate·입력 불변과 실패를 검사한다. 생성 CUDA 검사는 E317 공유 장치 검증 이후 별도 root 확인으로만 허용하며 이번 위임에는 포함하지 않는다.

현재 누적8506.463176899651/8560초와 예약0, E318의 미적용 미래 배정 제안, 기존 실패·M2·UI·미사용 F6를 유지한다. E317/E318/E319가 통과하면 같은 실제300control 전체에서 neutral 재현 및 이 shape loss의 유한 크기 방향 미분을 검사한다. 그 결과가 지지할 때만 시작 제한을 만족하는 작은 연결 변화의 제한된 학습을 실행한다. E24/E25의 장기 반올림 예측 실패와 E29의 mixed-loss 기각을 그대로 보존한다. 새 외부 위상/교사/controller/root 교정은 최종 runtime에 추가하지 않는다.


## Root SECOND 이후 생성 CUDA 확인

E319 CPU R1/SECOND와 E317 실제 생성 CUDA 수식 검산을 통과한 뒤, root가 생성300control의 동일 CPU/CUDA 수식·gradient·영점 RMS·resolved device를 비교한다. Hot evaluate/backward의 CPU tensor extraction을 금지한 검사도 포함한다. 단회15초/CPU1/host2GiB/additionalGPU256MiB, 원본/fit/신체0이며 이전 실패/비용/기준은 불변이다.
