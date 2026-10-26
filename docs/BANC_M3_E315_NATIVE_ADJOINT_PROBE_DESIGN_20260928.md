# E315: 실제 현재 BANC에서 국소 미분의 유효성

상태: 구현만 허용. E314 두 구현의 root source 승인 및 CPU/CUDA 생성 검증 후 root가 입력·소스를 동결하고 원본 60초를 예약할 때만 한 번 실행한다. 새 학습/후보 채택/실제 신체 보행 0이다.

## 질문과 결정

E314의 작은 생성 모델에서 수식이 맞아도 전체158706 신경·원래11584852 CSR에서 forward가 달라지거나 E24/E25처럼 형식 미분이 실제 float32 변화를 예측하지 못할 수 있다. 현재 E150 계열의 SSP/u0에서 (1) 처음20tick이 이미 저장된 native endpoint와 exact이고 (2) .5초 저장 상태의 짧은 실제 변화량을 국소 surrogate gradient가 예측하는지 확인한다. 통과는 장기 adjoint·근육학습·신체 미분 또는 gait 성공이 아니다. 실패하면 해당 학습 경로를 사용하지 않고 실패 원인에 따라 수치 구현/학습 방식을 재검토한다. 성공하면 원래 startup 제약을 포함한 한정된 native recurrent learning을 별도 동결한다.

## 고정 원본

- CSR: `e150-native-monotone-body/actual-global-csr.npz`.
- physiology: `e150-native-monotone-body/motor-timescale-configuration.npz`, E313의 `e313-load-model-parameters.npz`, 저장 neural metadata의 전체 `motor_timescale.inverse32`.
- neutral initial: `e305-ssp-first-control/neutral/initial-neural`, native 비교 endpoint: `e307-ssp-neutral-body/neutral/first-control-neural`.
- 이동 이후 상태: `e307-ssp-neutral-body/neutral/final`의 network (tick2000). 기존 노출된 개발 신체 상태이고 F6가 아니다. 신체를 생성/advance하지 않는다.
- mask: `e29-premotor-recurrent-robust-direction/recurrent-mask.npz`의 population/positions/edge_parameters. positions/population을 현재 `e289-recurrent-resource-roster.npz`와 exact 확인한다. E29의 source_weights를 현재 가중치로 쓰지 않는다. 원래 E150 CSR bytes만 사용한다. 그룹8547, premotor7136, 간선725849이며 새 edge와 부호 변경은 없다.

모든 경로는 `verification/banc-natural-gait-m3-20260919/` 기준이다. StateStore manifest를 검증하고 실제 사용 파일과 producer manifest/review를 동결 입력에 기록한다. 옛 실행 source와 현재 source를 혼동하지 않는다. 실제 source_identity, dtype/shape/finite 값·tau/gain/inverse, dt=.00025, cap200/coefficient1/200/threshold7.5, mask 및 tick을 확인한다. GraphStore·native body 생성은 필요 없다.

## 고정 계산

1. GroupedCurrent/SSPAdjointModel 한 쌍을 구성한다. theta1로 initial rate/z에서 saved first-control drive를20tick 고정하여 진행한다. 최종 전체 r/z와 mask/drive가 native 저장 비교와 exact여야 뒤 계산을 허용한다. 원래 native를 재실행하지 않는다.
2. 이미 저장된 .5초의 r/z/drive/mask에서 theta1로4tick(.001초)을 functional autograd로 진행한다. 손실은 최종7136 premotor rate의 **float64 평균**이다. 이 손실은 미분 검사용이며 보행 품질의 목적함수가 아니다. full-state adjoint를4tick 모두 전달하고 효능 gradient를 저장한다. physiology와 시작 상태는 학습하지 않는다.
3. 방향은 사전에 `d=sign(gradient)`로 정한다. 기존θ=1에 `±.001*d`를 float32로 한 번 변환하여 각각 동일 .5초 상태/같은 drive/mask에서4tick씩 진행한다. 격자·라인서치·solver·가중치 선택·후보 export는0이다. 전방/후방 중 좋은 결과를 고르는 실험이 아니다.
4. 실제 loss차 `Lplus-Lminus`와 gradient·`(theta_plus32-theta_minus32)` 내적을 비교한다. 양자가 finite이고 predicted 차와 observed 차가 각각1e−7보다 크며, 같은 부호/상대오차≤.05여야 국소 screen을 통과한다. 0변화 또는 반올림으로 사라진 방향은 통과하지 않는다. 오차 기준은 float32 변수의 크기를 본 뒤 바꾸지 않는다.

과학 forward 합계 최대32tick(.008모델초): prefix20+nominal4+plus4+minus4. 역미분4tick은 별도 기재한다. 재실행/warmup/capture는 수행하지 않는 functional 경로이며 생겼다면 별도 계수한다. 저장값 참조는 모델 advance로 세지 않는다. 중립 실패 이후 변경된 후보 계산은 실행하지 않는다.

## 실행·검토

actual gpt-6-sol/xhigh가 도구·생성 fixture test 두 파일을 구현하고 root FIRST→같은 Sol 필요시 첫 수정→SECOND→root 잔여 수정 절차를 따른다. 도구는 `--freeze PATH --output PATH` 입력, 새 출력만 생성하며 hash/qualification을 먼저 검사한다. 원본 제한은 원본1/자동재시도0, internal science45초+finalization5초/감독60초, CPU1/GPU1/host6GiB/추가GPU2GiB/출력200MiB/free2GiB. root의 기존 supervisor를 사용한다. 비용·partial/failure output·실행 source를 남기고 failed/NaN/timeout을 음성 효과나0비용으로 바꾸지 않는다.

현재 원본누적8495.160237499575/8560초·잔여64.83976250042542초다. 실행 gate를 통과할 때60초를 같은 배정 내에서 예약하며 증액0이다. 생성 CPU fixture검증은 agent45초/root45초, CUDA생성 검증은 E314의 공유60초에 포함한다. 단일GPU를 공유하고 백그라운드 서버/UI·M2·F6·F1–F8·이전 실패는 보존한다.
