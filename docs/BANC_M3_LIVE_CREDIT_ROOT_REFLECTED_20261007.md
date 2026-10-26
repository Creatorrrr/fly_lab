# Astra 검토 반영: 저장된 실제 입력의 출력 대응부터 검증

ROOT가 actual GPT-6 Astra/max 리뷰 JSON19274B/SHAe7c3e09ec24294eefc0a5f09d6435c33415cbf7128f6123978cc282e3f77830b와 한국어 MD13665B/SHA102ee6cacf6c585b462e379abe09d95dbe946f54d40409e4dbce3bacd289dafa를 읽고 R1–R5를 반영했다. 원형 ROOT 설계는 보존한다. verdict REVISE_BEFORE_SOL_IMPLEMENTATION을 수용하며 새 인간 승인 대기 없이 아래의 더 작은 FIRST를 진행한다.

## 지금 구현하고 실행할 범위

먼저 **saved replay만 한 번** 구현한다. 이미 완료된 tracking 두 arm 각각4개 pre-state와16 actual finalized drives로 arm당16controls, 합32controls/640neuralticks를 실행하고 command42/tendon6만 실제 저장값에 대조한다. 별도 physics/body/geometry 실행·gradient·새 candidate·weights·alpha 생성은0이다. 추가 body300/CPU120은 이번 FIRST 범위에 넣지 않는다. replay 결과를 보고 ROOT가 기하 자료의 필요성을 판단한 이후의 조건부 다음 단계로 보존한다. 이번 단계는 출력 대응이며 live gait credit/VJP 검증으로 부르지 않는다.

각 arm의 control starts100/164/228/292와 ends101..104/165..168/229..232/293..296를 exact 바인딩한다. rate/adaptation float32[4,158706], bank float64[4,391,3], original_activation float64[4,96], resource float64[4,7136], 실제 drives float32[16,158706], ordered motor391/source7136/output mask와 u0/resource1을 쓴다. dt .00025/20ticks/control, decoder dt .005를 유지한다. 각 window의 해당 arm 상태와 입력을 복원하고0으로 리셋하거나 arm 간 교환하지 않는다.

기존 SSPAdjointModel/MonotoneDecoder/4-control CheckpointedControlRollout와 최소 assembly를 재사용한다. neutral에는 원본 deployed float32 CSR, candidate에는 정확한 selected deployed float32 weights를 사용하고 multiplier theta=1로 replay하여 theta를 이중 적용하지 않는다. candidate archive identity는 기존 θ/selected weights SHA로 묶는다. historical endpoint를 필요로 하는 전체 coordination runner, body checkpoint/constructor, 별도 native comparator를 호출하지 않는다. warmup/capture/gradient/recompute/backward는0이며 총량640 안에서 실제 requested/returned/device-confirmed work를 보고한다. 추가 비교 구현을 몰래 실행하지 않는다.

관측된 command/tendon 비교는 기존 deployment reproduction 최대 절대오차1e-11/rtol0으로 사전 고정하고 exact/max/RMS를 함께 보고한다. 입력의 bytes/shape/dtype/ordered IDs/bitmaps/clock identity는 tolerance0이며 clock은 기존2e-8s 기준이다. 미저장 post-window rate/adaptation/readout/bank/original_activation endpoint는 NOT_OBSERVED로 남긴다. 출력 대응 성공은 전체 신경 상태 동치나 생물학적 신경 메커니즘 증거가 아니다.

## 기하 후속 설계의 필수 교정

기존 body.step_joint_targets는50 native steps 뒤 기존 _forward_physics를 호출한다. control-boundary foot/thorax/frame은 그 기존 forward 뒤 post-qpos/qvel과 묶는 것이 기본이다. extra live forward0 조건은 기존 forward를 제거한다는 뜻이 아니다. last-native-return cache/tail-pre pose는 별도 채널이며 그 채널의 cache 시점은 실제 integrator와 호출 경로로 사전 입증해야 한다. pre/post 중 오차가 작은 자료를 사후 선택하지 않는다.

추가 기하가 필요하면 실제 초기화·restore 후 compiled MJB 한 번과 bytes/SHA/MuJoCo version/dimensions/integrator/ordered joint/qpos/dof/actuator/tip/thorax IDs·names·mm/rad/mocap/변경 model 상태를 저장한다. archive와 serialization/supervision을96MiB combined cap에 포함한다. 별도 model/MjData에서 full qpos/qvel과 필요한 mocap를 복원하고 mj_forward만 사용한다. 새 FlyGymBody나 live data를 재사용하지 않는다. geometry 관련 model 불변을 확인한다.

고정 검증은 position1e-9mm/Jacobian1e-8mm/rad/centered active hinge FD±1e-6rad/rtol0다. thorax frame은 원점과 원점+1mm*axis 세 점의 위치로 검증한다. Jacobian 열은 dof IDs, perturbation은 hinge qpos IDs이며 root/passive/model과 시작 프레임을 고정한다. free kinematics FD에는 ctrl clamp를 적용하지 않고 reachable-control 분석을 별도로 한다. tip은 tarsus5 body origin이며 pad 표면의 관측점이 아니다. vertical은 world+Z다.

최소 attribution은4 anchors 각각 c→c+1의 전이만 다룬다. J_A(q_c)*(effective_ctrl_(c+1)-q_A,c), J_A(q_c)*(q_A,c+1-q_A,c), 관측 tip delta를 구분한다. active command의 effective 값은 compiled actuator clamp이며 joint constraint range와 구분한다. world displacement를 고정 시작 몸통 프레임으로 투영한 혼합 잔차와 각 시점 자기 몸통 프레임의 body-relative AP를 혼용하지 않는다. rigid root 운동은 body-relative AP에서 상쇄되므로 그 잔차를 root 원인으로 명명하지 않는다. loaded/unloaded/transition은 같은 두 경계의 mask·분모를 쓰고 빈 범주는 UNOBSERVED다. 네 국소 J를200control stance 전체로 확대하지 않는다. 이 projection은 힘·추진·servo/미래접촉 인과가 아니다.

## 실행, 실패와 다음 판단

지금은 Original single role live_replay outer240/child220/internal200/cleanup20초, host24GiB/output16MiB/GPU delta+allowance8GiB/sample3000만 준비한다. 같은 J grant/offset/carry와196debits, science1287.124/6000, stage1413.485/6300, global11307.239960899506/16200, generated142.378/300을 유지한다. 증분grant0/worker1/자동재시작0이다. 새 Original 역할에는 closed tracking3의 정확한 입력 manifest/context/control/body-report와 비용을 연결하고 기존 supervisor/once writer의 작은 routing만 재사용한다. 범용 gate/certificate 계층을 만들지 않는다. generated 잔여에서 source/lazy import/ordered state/window/deployment/count/output 및 중복한번 기록의 의미 있는 검사만 수행한다. 기존 통과한 불변 tests를 새로 확대하지 않는다.

실제 mismatch나 실행 실패는 부분 출력과 once 비용을 보존하고 해당 구현의 차이를 ROOT가 해석한다. 성공 cache, 사후 tolerance 완화나 같은 call 자동 재시작은0이다. 출력 대응 실패 시 추가 physical capture보다 대응 경로의 차이를 먼저 수정한다. 성공 시 기존 RH loaded AP association과 command geometry 부재가 다음 변경을 결정하는 데 충분한지 ROOT가 판단하고, 필요한 경우만 이미 검토된 body300+CPU120의 후속 범위를 진행한다. 최대660초는 유지하되 사용한 실제 비용만 청구한다.

실제 GPT-6.1 Sol/xhigh에 작은 Source-only FIRST를 맡긴다. sparse packed forward/recorded live state의 정확한 대응과 기존 원장·Source guard 연결의 난이도를 고려한 effort다. FIRST stable proposal 후 ROOT FIRST→필요한 같은 Sol sole R1→ROOT SECOND→ROOT 잔여/실제 적용과 실행 순서를 유지한다. 모든 실행은 아직 NOT_RUN이다. F1–F8 ACTIVE/INCOMPLETE, 기각 후보 채택false, M2/UI/provenance·미사용 F6 노출·biology UNPROVEN과 고정 gait thresholds를 보존한다.
