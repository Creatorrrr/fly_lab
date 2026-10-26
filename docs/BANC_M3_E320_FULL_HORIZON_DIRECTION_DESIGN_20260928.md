# E320: 현재1.5초 신경 경로와 운동 형태 loss의 실제 방향 미분

E315의1ms 평균 premotor 미분 통과는 실제 보행 학습 구간의 신뢰성을 보장하지 않는다. E24/E25의 장기 미분 실패를 보존하고, 현재 SSP/u0 및 고정 E150 decoder에서 처음0–1.5초 전체를 연결하여 확인한다. E318의 실제 full drive300 수집과 root 원자료 감사가 통과하기 전에는 원본 실행하지 않는다. 이번 단계는 한 방향의 판별이며 optimizer 반복·보행 후보 채택·신체·F6는0이다.

H320은 현재 전체 BANC의 기존 P→P 연결을0.1% 바꾸었을 때, E317의 full BPTT가 E319 shape loss의 유한 차이를 정해진 정확도로 예측한다는 것이다. 경쟁 설명은 긴 구간의 f32 반올림/비선형성 또는 잘못된 고정-input 재현 때문에 국소 gradient가 실제 효과를 예측하지 못한다는 것이다. Native 재현 실패는 구현 유효성 실패이며 학습 가능성의 반증과 구분한다.

## 구현과 고정 입력

Actual gpt-6-sol/xhigh가 신규 `tools/verify_c_banc_m3_full_horizon_direction.py`, `tests/test_c_banc_full_horizon_direction.py`만 구현한다. Root FIRST → 필요한 같은 Sol R1 → root SECOND → root 잔여수정을 지킨다. 생성CPU 각Sol/root120초·CPU1/host2GiB, 원본/신체/CUDA/fit0. 기존 E314/E316/E317/E319 및 source는 수정하지 않는다. 함수는 작은 주입된 fake dependencies로 검증할 수 있어야 한다.

신경 구성은 E315가 실제 검증한 current158706/11584852 CSR와 membrane/adaptation 개별 모수, coefficient=.005/cap200을 그대로 재사용한다. E29의 mask는725849개 기존 P→P edge position과8547 group ID로만 쓰고 당시 source_weights는 사용하지 않는다. 모든8547 효능의 시작은1이며 다른 가중치·생리·decoder·mask는 고정한다. E315 load_inputs의 현재 원본 검증을 재사용할 수 있으나 E315 원본 실행 함수는 호출하지 않는다.

Initial r/z/mask 및 bank/original activation, motor order와 decoder calibration은 E318 observed-start whole에서 가져온다. E315 initial neural과 일치해야 한다. E318300개 실제 full f32 drive를 원래 순서와20substep씩 유지한다. 과학적 body feedback을 재실행하지 않는 **조건부 신경 입력**이라는 한계는 결과에 명시한다. CSR 및 static float32 신경수식, same control order, 고정 `.005` decoder, startup99 원래 판정을 보존한다.

E319의 curve는 기존 `e85-space-compatible-cadence-reference/commands.npz`에서 E107의 `reference_at_phase`와 같은 주기 보간으로 한 번 만든다. 원래 reference의 time≥5초와 LF phase column0을 사용하여 phase `arange(128)/128`에서42개관절을 보간하고 (6,128,7)로 나눈다. 이는 기하학적 닫힌 곡선이며 절대 시간·teacher 주파수·다리 간 phase를 강제하지 않는다. 피험자 결과를 보고 M/구간/곡선을 선택하지 않는다. 원래 `e103-startup-preserving-recruitment/startup-inputs.npz`의 target_command/target_tendon99개를 E319 startup reference로 쓴다. E318 자체 q를 startup 정답으로 대체하지 않는다.

## 단일 판별

1. E317 checkpoint_controls=1, steps_per_control20로 control1..300을 theta1에서 한 번 진행한다. 가능하면 외부 관측 callback 없이 전체 q/LTM와 final r/z/bank/original을 반환한다. 최종 r/z는 E318 control300의 실제 full endpoint와 dtype·bytes 동일해야 하고, 300개 q/LTM는 최대차≤1e-11rad/명령, 최종 bank/original은≤1e-10이어야 한다. 최초 상태 및 control300 tick/identity/모든input provenance도 일치해야 한다. 전체 q의 재현은 모든 제어를 포함하며 endpoint 신경 exact를 모든 substep exact로 확대하지 않는다.
2. E319 shape_loss만을 theta에 한 번 역미분한다. 초기0–.5초도 gradient 경로에 남으며 truncated BPTT나 중간 state detach는 없다. Gradient 전체8547을 저장하고 유한·비영인지 검사한다. 영값 또는 비유한이면 판별 불충분/실패를 기록하고 나머지 방향 실행을 하지 않는다.
3. 방향은 `sign(gradient)` 하나로 고정한다. 실제 theta±는 `f32(1 ± .001*direction)`를 f64로 정확히 표현해 엔진에 전달한다. 두 조건은 각각 **동일 control0 전체 초기 상태**에서 시작하고 같은300 drive를 사용한다. theta 변화는 처음부터 일정하며 .5초에 켜는 switch가 아니다. 새 방향·step-size grid·재시도·branch 선택은 없다.
4. `observed=Lplus−Lminus`, `predicted=sum_f64(g*(theta_plus−theta_minus))`. 둘 다 유한이고 절대값>1e-9, 부호가 같고 `abs(observed−predicted)/abs(predicted)≤.05`일 때 이 한 방향/크기를 supported로 판정한다. 작은 값은 inconclusive, 유한하고 기준 미달은 refuted다. Minus loss가 neutral보다 감소하고 기존 startup4제약을 만족하는지도 별도 기록하되, 이 조건만으로 body/F3–F8나 생물학적 보행을 승인하지 않는다. Plus와 minus 양쪽 loss·startup·perleg·q/LTM·최종 상태·효능 및 실제 f32 변경된 선택가중치 수/범위를 모두 남긴다.

Curve projection의 최근접 선분이 바뀌는 비매끈함도 실제 차이에 포함된다. 이 사전 기준을 실패 후 완화하거나 shape loss를 다른 proxy로 교체하지 않는다. Neutral 재현이 통과하고 방향 예측도 지지하면 같은 작은 minus 효능을 시작하는 후보의 native startup/short를 다음 별도 계약에서 검사한다. 실제 body에서 나빠지면 조건부 loss 개선으로 덮지 않는다. 결과가 미지지하면 긴 gradient를 학습 근거로 쓰지 않으며 어떤 경계에서 깨졌는지 현재 원자료로 구분한 후 방향을 재설계한다.

## 회계·출력·실행 경계

E317 receipt의 requested/host-returned/device-confirmed forward/recompute/backward를 각 호출에서 구분한다. 실제 CUDA synchronize/host copy 이후만 confirmed로 기록한다. Baseline backward 전후 및 각 방향 반환에 raw를 바로 보존하고, 실패한 receipt·첫 오류·이후 저장/cleanup 오류를 남긴다. Gradient-free 두 방향은 backward0이다. 예상 상한 forward18000+recompute6000tick, backward6000tick에 해당하는300block이며 새물리/신체/optimizer0이다. 생성·kernel compile·계수 준비 시간을 과학적 tick으로 세지 않되 전체 wall에 포함한다.

원본 run은 별도 source/input/qualification pin과 root review 후 단회·retry0이다. 예상 감독480초/internal450초/science410초/finalization40초, CPU1/host6GiB/additionalGPU8GiB/output2GiB/free4GiB로 제한한다. 실제 total device memory 변화와 Torch peak를 분리하고 source/input/output을 pin한다. 현재는 원본 예약/추가 배정0이며 E318의150초 예약도 아직 검토 중이다. 현재 누적8506.463176899651/8560초와 모든 이전 비용을 유지하고, 원본 실행 직전에 필요한 다음 단계 배정만 사전 기록한다.

실제입력dataset·E317/E319 generated qualification·E315 source validity를 검사하는 passive gate는 heavy model/CUDA allocation을 하지 않는다. Root가 원자료에서 observed/predicted와 startup/error, 비용을 별도 재계산한다. F1–F8/M2/활성 UI/F6미노출/과거 실패는 불변이다. 본 단계 완료는 학습을 위한 판별의 진전이며 자연 보행 목표의 완료가 아니다.
