# 기록 상태에 따른 지지 발 운동과 신경 credit 검증

이 문서는 실제 추종 계측 이후 ROOT의 다음 설계다. 사용자의 최신 지시대로 추가 승인을 요청하지 않고, ROOT 설계→actual GPT-6 Astra/max review→ROOT 반영→actual GPT-6.1 Sol 적정 effort FIRST→ROOT FIRST→같은 Sol sole R1 필요 시→ROOT SECOND→ROOT 잔여 수정 순서를 따른다. 아직 새 단계의 실행이나 채택은 없다.

## 목적과 결정

보행 실패를 실제 감각 입력을 받은 신경 상태와 발의 국소 운동에 연결할 수 있는지 검증한다. 현재 고정 후보를 다시 튜닝하기 전에, 학습에 쓰는 local credit가 실제 실행의 명령과 물리적 발 방향에 대응하는지 확인한다. 이 유한 단계의 성공은 대응 자료와 검증된 분석 경로 확보이며 F1–F8 보행 성공이나 생물학적 기전 입증이 아니다.

실제 tracking 결과 두 arm은 common historical trajectory가 exact=True였고 요청−실제 관절 RMS 약 .00156 rad, native effective−postq RMS 약 .00388 rad다. 후보의 실제 전진1.60588mm, slip1.96050mm/s, bout.21s로 계속 기각된다. RH의 지지기 body AP는 다른5다리와 반대 부호지만 원인은 미식별이다. 게인 증가보다 recorded-state kinematics와 actual-drive neural replay 대응을 우선한다. 큰 전달 오류가 없다는 확정이나 RH 한 다리의 원인 확정은 하지 않는다.

## 좁은 실행 순서

1. SOURCE audit와 generated fixture로 compiled active42/qpos/velocity IDs, hinge/unit gear, leg/tip/frame/mm/rad와 동일 모델 자료의 저장·복원 경로를 고정한다. 기존 decode는 original activation/adhesion을 업데이트한 뒤 monotone bank를 반영한 target42를 내며, LTM/singleton 고정은 기존 계약이다. 이 구조를 임의 버그로 간주하거나 바꾸지 않는다.
2. 기존 E150 neutral의 동일0→99→300 실제 인스턴스 한 번만 실행한다. 기존 continuous arm/초기 증명/정상 감각→neural20→decoder→physics50을 그대로 유지하면서 full qpos/qvel과 실제 tip 좌표·몸통 프레임을 저장한다. 각 control 마지막 native step 직전의 full qpos도 저장한다. MuJoCo cached xpos와 integration 후 qpos의 시점을 혼동하지 않도록 pre/post clocks·integrator를 기록한다. 추가 계측은 복사·읽기만 하며 live mj_forward/step/추가 encode·decode/신경 snapshot을 넣지 않는다. 기존 tracking neutral의 common control/native/trace와 exact/max/RMS를 비교한다. mismatch는 새 trajectory로 명시하고 사후 epsilon 또는 자동 재시작으로 통과시키지 않는다.
3. 저장된 compiled model과 full pose를 별도 MjData에서 복원해 tip/body frame을 검증한다. 4개 고정 boundary100/164/228/292에서 active42에 대한 tip Jacobian을 구한다. root/passive를 고정한 centered finite difference ±1e-6 rad로 검증한다. 전체 body 기하와 passive 값을 그대로 쓰며 frozen-neutral TorchFootGeometry로 대체하지 않는다. mj_forward만 실행하고 동역학 step은0이다. cached tip은 마지막 native pre-state에서, post-integration q의 기하는 post-state에서 각각 검증·표시한다. freshness나 동일 모델 재현 실패는 geometry UNAVAILABLE로 종료한다. 미리 고정한 위치/Jacobian 허용오차는 기존 기하 검사와 모델 단위를 근거로 Astra 검토에서 확정한다.
4. 이미 저장된 tracking 두 arm의 pre-state4/actual drive16으로 조건부 neural replay를 검증한다. ordered motor391/source7136/output mask와 u0/resource1, 실제 float32 deployed weights/theta, dt와20 neuralsteps/control을 보존한다. arm마다4×4 controls, 전체 replay32controls/640neuralticks만 허용하고 gradient는0이다. 고정된 같은 arm의 controls101..104/165..168/229..232/293..296 command42/tendon6과 대조한다. source/decoder/신경 endpoint별 기존 deployment tolerance를 사전에 명시하고 mismatch를 동치로 숨기지 않는다. 다른 arm의 actual drives를 섞거나 초기 상태를0으로 재설정하지 않는다. 결과는 recorded-drive conditional replay이며 새로운 closed-loop 보행이나 미사용 초기상태 검증이 아니다.
5. 대응이 확보되면 실제 loaded/unloaded/transition mask와 Jacobian으로 active-joint 변화의 AP/vertical 국소 기여를 분리한다. compiled control clamp를 반영하고 joint range와 구분한다. 기여는 순간 선형 민감도이며 힘·추진력·future contact 인과가 아니다. 실제 foot delta에는 floating-root/passive/nonlinear motion도 포함되므로 전부 active credit로 귀속하지 않는다. 이 단계에서 새 weights/candidate/alpha를 생성하지 않는다. 후속 방향은 stance에서 backward foot motion, swing clearance와 다리 전환·지지 보존을 다룰 수 있는 범위로 ROOT가 실제 증거를 보고 설계한다.

## 입력과 불변 조건

`parent-tracking-ROOT-actual-result-20261007.json`, 두 capture manifest/body-report/control/native/trace/tracking/context, 실제 qualification c와 source freeze c, J grant/input/selected candidate/audit와 E150 neutral provenance를 연결한다. 신경 해부학적 node/edge/sign, anatomy moment/decoder 계수, body 설정·마찰·근육/adhesion/tendon, UI/M2, 기준과 실패를 보존한다. 원본 파일을 재생성하거나 덮어쓰지 않는다. 이번 확인으로 사용한 pose는 F6 미사용 pose로 다시 주장하지 않는다. teacher/reference는 분석에만 쓰고 최종 실행 외부 CPG/phase/trajectory/root 교정은0이다.

## 유한 작업과 자원

기존 J grant 하나와196개의 실제 debit, generated142.378초, science-local1287.124초를 이어받고 추가 grant0/worker1/자동 재시작0이다. 한 neutral body outer300/child280/internal250/cleanup20초, host24GiB/output96MiB/GPU delta+allowance8GiB; 별도 saved kinematics CPU outer120/100/80/20초/host2GiB/output16MiB; recorded-drive CUDA replay outer240/220/200/20초/host24GiB/output16MiB/GPU8GiB다. 총 Original outer660초가 현재 science 잔여4712.876초 및 stage/global 잔여 안에 있다. 기존 generated 지역 잔여157.622초에서 필요한 fixtures/one-owner checks만 실행한다. 결과·직렬화·supervision streams의 combined cap을 지키고 failures/timeout/unknown를 once 비용과 함께 남긴다. 기존 J8/saved/tracking의 guard와 역사적 cap은 확장하지 않는다.

별도 작은 role routing/roster와 직접 실행 경로를 재사용한다. 정책/승인 metadata를 늘리는 것이 성과가 아니며 outcome은 actual model/kinematics/replay 일치, 불일치 원인과 다음 scientific 방향이다. 진단이 실패하면 부분 자료를 보존해 특정 구현 문제를 수정하고, 같은 call을 재사용하거나 실패 기준을 낮추지 않는다. 장시간/10초/F6/F7 검증은 실제 short candidate가 고정 기준을 통과한 이후다.

## 완료 증거와 다음 선택

필수 증거는 frozen Source/실제 tests, known process close/once 비용, actual body continuity와 과거 값 비교, pre/post full pose와 동일 compiled model 재현, Jacobian finite difference, 각 arm32controls 안의 native deployment/decoder 대응, 제한·미식별과 다음 방향을 포함한 한국어 보고서다. H1/H2를 강제로 분류하지 않는다. 새 후보 채택은false, F1–F8 ACTIVE/INCOMPLETE, biology UNPROVEN이다.
