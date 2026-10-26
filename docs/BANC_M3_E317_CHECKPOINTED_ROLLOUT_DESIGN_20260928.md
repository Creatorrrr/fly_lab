# E317: 전체 SSP와 근육 변환의 구간 재계산 미분

현재 E314/E315의 전체 신경 국소 미분과 E316의 근육 변환을 한 주기 이상 연결하는 구현 단계다. E150에는 반복 운동이 있으나 지속 전진이 끊긴다. 목적은 그 주기 안의 근육 동작을 학습할 계산을 준비하는 것이며, 메모리 검증이나 loss 감소를 자연 보행으로 판정하지 않는다. F1–F8, 기존 실패·비용, M2·활성 UI·미사용 F6는 유지한다.

H317은 제어 시점의 상태만 저장하고 역방향에서 구간을 재계산하면, 동일한 native forward와 전체 길이 gradient를 유지하면서 모든 중간 그래프를 동시에 보존할 필요가 없다는 것이다. 경쟁 설명은 상태 경계 누락, 마지막 cotangent 누락, shared cache 변조 또는 재계산 순서 차이로 결과가 달라지는 경우다. Truncated BPTT와 상태 detach에 의한 경계 gradient 차단은 허용하지 않는다.

## 구현 계약

소유 파일은 신규 `tools/banc_m3_checkpointed_rollout.py`와 `tests/test_c_banc_checkpointed_rollout.py` 두 개다. E314·E316 엔진과 제품 경로는 수정하지 않는다. Root 설계 → actual gpt-6-sol/xhigh 구현 → root FIRST → 필요한 같은 Sol R1 → root SECOND → root 잔여 수정 순서를 따른다. 이번 위임은 생성 CPU 구현·검사만 허용한다.

API는 `CheckpointedControlRollout(neural, decoder, motor_ids, *, steps_per_control=20, control_dt=.005, checkpoint_controls=1, max_controls=320)`와 `.run(rate, adaptation, bank, original_activation, drives, efficacy, mask)`다. Neural은 E314 인터페이스이며 decoder는 E316이다. Constructor는 motor IDs를 소유 복사하고 정렬된 유일 391개, 범위와 장치, 양의 정수 한도, 신경 dt × steps와 control_dt의 float32 반올림을 고려한 일치를 검사한다. 잘못된 bool/float 정수는 거부한다. 모델/고정 계수는 그래프 생존 중 변경할 수 없다. 해당 조건은 문서에 명시하고 native 객체의 tensor identity/version 변경도 검출한다.

입력 r/z는 (N,) 연속 float32, bank는 (391,3), original activation은 (96,) 연속 float64다. 고정 drives는 (T,N) 연속 float32, mask는 (N,) binary float32이고 둘 다 requires_grad=False다. Efficacy는 동일 장치의 양수 연속 float64 group vector다. T는 1..max_controls다. 모든 입력은 유한하고 신경·decoder 상태는 해당 엔진의 유효 범위 안이어야 한다. API는 입력을 변경하지 않는다.

결과 dict는 `q` (T,42), `tendon` (T,6), `rate`, `adaptation`, `bank`, `original_activation` 최종 상태와 호출마다 별개인 `receipt`다. q와 tendon 및 decoder 상태는 float64, 신경 상태는 float32다. Activation 전체 기록은 이 API의 출력이 아니다. 초기 상태와 efficacy의 first-order gradient를 지원한다. drive/mask/고정계수와 higher-order 미분은 지원하지 않으며 higher-order 요청은 명시적으로 거부한다.

Custom autograd forward에서 no_grad로 동일한 신경 substep → ordered MN readout → decoder 순서로 실행한다. Native MN readout은 raw `rate[motor_ids]`이며 mask를 다시 곱하지 않는다. Mask는 presynaptic current에만 적용한다. 제어 구간 경계의 네 상태를 별도 소유 checkpoint로 보존한다. 준비된 efficacy는 trajectory마다 한 번 만든다. 역방향은 block을 역순으로 순회해 경계 상태를 gradient leaf로 만들고 동일 block을 재계산한다. 해당 block의 q/tendon upstream과 다음 경계의 네 cotangent를 함께 전달하여 `autograd.grad`로 상태와 efficacy gradient를 계산한다. Efficacy gradient는 float64로 누적한다. Backward 시작 때 별도 leaf/cache 준비 한 번은 허용하며 hot block/step에 CPU scalar 추출·복사·동기화를 추가하지 않는다. 원래 장기 반올림 surrogate 한계는 유지한다.

원본 입력을 save_for_backward로 보존해 입력의 in-place 변조를 검출하고, 두 개의 살아 있는 호출이 checkpoint/cache/receipt를 공유하지 않도록 한다. 사용하지 않은 출력의 None cotangent와 최종 상태만 사용하는 loss, 모든 q/tendon 사용, uneven final block을 지원한다. Backward 중 실패를 완료라고 기록하지 않는다.

Receipt는 forward/recompute의 neural tick과 control에 대해 요청/host-returned 수를 구분한다. Backward는 block 요청/반환 수 및 반환한 block에 속한 tick 수를 기록하며 실패한 block의 개별 역미분 tick 수를 추정하지 않는다. `receipt.confirm()`이 해당 CUDA 장치를 실제 synchronize한 뒤 확인된 host-returned prefix를 기록한다. 확인 전 device-completed 수는 미확인이다. CPU도 동일한 명시적 확인 인터페이스를 쓴다. 호출·역미분 wall은 필요하면 외부 감독에서 계측한다.

## 판별과 자원

생성 CPU는 작은 독립 differentiable fake neural과 실제 E316 decoder를 써서 재계산 경계와 chain rule을 검증한다. Naive full unroll 대비 모든 forward는 byte equality, 모든 지원 입력 gradient는 상대 L2 ≤5e-6 또는 최대 절대 ≤1e-8이다. 작은 독립 dense 수식과 directional FD도 사용한다. 두 live graph, frozen input/parameter/model mutation, unused/final-only loss, 부적합 입력과 실패 후 receipt, checkpoint 폭 1과 비정수 마지막 block을 확인한다. 현재 E314의 Python CPU sparse loop 때문에 큰 실제 신경 fixture를 반복하지 않는다.

Root FIRST 통과 뒤 real E314+E316의 생성 391-cell CUDA rollout과 unroll 및 독립 dense reference를 비교한다. 같은 경계·forward·gradient 기준을 적용하며 hot step의 CPU 호출 금지와 저장 tensor 수/peak memory를 확인한다. Generated의 속도·메모리는 실제 158706-cell 장기 학습의 성능으로 일반화하지 않는다.

Sol CPU 전체90초, root CPU 전체90초, 각 CPU1/host2GiB. Root 승인 후 생성 CUDA 전체60초/host2GiB/additional GPU1GiB/출력100MiB. 생성 seed31701. 원본 allocation/load/신경/신체/fit/F6는 모두0이며 원본 예약도0이다. 현재 원본 누적8506.463176899651/8560초·잔여53.536823100348556초를 유지한다. 실제 full-drive 수집과 full-size 방향 검증은 각각 별도 E318 및 후속 계약이다. 실패와 비용을 삭제하거나 초기화하지 않는다.
