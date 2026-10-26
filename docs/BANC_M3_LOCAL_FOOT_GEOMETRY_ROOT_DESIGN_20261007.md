# 실제 국소 발 기하로 다음 신경 credit를 결정하는 설계

Finite-goal은 BANC 자체의 지속 보행이며 고정 F1–F8은 ACTIVE/INCOMPLETE다. 최신 사용자 지시는 승인 대기 없이 범위 안에서 판단해 진행하라는 것이다. Workflow는 ROOT 설계→실제 GPT-6 Astra/max 검토→ROOT 반영→GPT-6.1 Sol 적절한 effort FIRST→ROOT FIRST→필요시 같은 Sol의 이번 단계 sole R1→ROOT SECOND→ROOT 잔여 수정/실제 실행이다. 이는 완료한 live replay 단계의 추가 R1이 아니라 새 physical diagnostic 단계다.

## 입증한 상태와 필요한 결정

기존 고정 후보 α=.0001/index4는 actual gait에서 기각됐다. Neutral/candidate forward1.6857973223/1.6058769432mm, slip1.9427288085/1.9605032380mm/s, longest moving bout .21초, qualified cycles0이다. Cycle0은 .3초 bout 부재 때문이며 다리 무운동의 증거가 아니다. 요청−측정 q RMS~.00156rad는 큰 전체 전달 실패를 약화하지만 인과 원인을 식별하지 않는다. RH loaded AP의 다른5다리와 반대 부호는 association이다.

실제 CUDA saved-state replay32controls/640ticks는 command 최대2.55351295663786e-15rad/tendon2.7755575615628914e-17로 fixed abs1e-11/rtol0를 통과했다. Bit exact false, post-state/MN/adhesion NOT_OBSERVED, VJP/live credit UNVERIFIED다. `verification/banc-natural-gait-m3-20260919/parent-live-replay-ROOT-actual-result-20261007-b.json`8242B/SHA7935721f587445b0a118fa0bb0ad8a70599f12da6f6cfaa826b66dd8cdb31aab와 actual report/once costs를 우선 읽는다. 초기 ROOT metadata의 잘못된 비용 키는 b에서 교정했으며 실제 원장·비용은 처음부터 올바르다.

다음 신경 방향을 설계하려면 실제 자세에서 command가 발 AP/vertical 이동으로 어떻게 이어지는지 필요하다. 관절각 오차만 줄이는 목표는 gait를 보장하지 않았다. 새 candidate/gradient를 만들기 전에 이미 Astra가 검토한 conditional neutral body300+CPU120을 진행한다. 기존 exposed neutral startup/body/params만 쓰고 미사용 F6는 노출하지 않는다. 이전 `docs/BANC_M3_LIVE_CREDIT_ROOT_REFLECTED_20261007.md`와 실제 Astra review의 시간/기하 교정을 읽고 이 문서의 좁은 범위로 구현한다.

## 두 Original 역할과 자원

`geometry_capture`1회: outer300/child285/internal270/cleanup15초, host24GiB, combined output96MiB(결과93+supervision3), GPU sampled delta+allowance8GiB/sample3000. 기존 factory/initial_neutral/0→99→300 동일 인스턴스/LiveObserver/tracking를 재사용한다. Constructor neural320/capture160/settle2000와 정상6000neural/15000nativephysics 등 실제 작업을 분리 기록한다. 보행제어/teacher/CPG/root correction/gradient/newcandidate 변경0.

`geometry_analyze`1회: outer120/child100/internal90/cleanup20초, host2GiB/output16MiB(결과13+supervision3), CPU only. 저장 MJB 한 개로 별도 MuJoCo model/MjData를 만들고 FK/Jacobian/FD만 계산한다. Torch/CUDA/neural/FlyGymBody construction/mj_step0. Actual Original 자료를 읽으므로 실제 비용을 Original lane에 한 번 청구한다.

Worker1/restart0, 역할 cap 합420초, 같은 J grant/offset165/새grant-reset0이다. 현재200debits/82directcalls, global11325.160960899506/16200, stage1431.4060000000002/6300, science1305.045/6000, generated155.67600000000004/300, sharedgenerated883.8880156000001/1700, jobs/reservations/pending/holds0을 보존한다. 신규 generated qualification의 합 outer cap≤110초(닫기 여유20 포함)를 계획해 잔여144.324초 안에서 필요한 source/import/small kinematic/changed-owner 검사만 수행한다. 기존 통과 불변 suites/과거 전체 E360 반복0. 실제 genuine full-return 비용만 기존 Native once writer로 청구하며 failed candidates/실패/누적 비용은 불변이다. 새 범용 gate/validator 계층을 만들지 않는다.

## 계측 시간, full state와 model

Production boundary는 body.step_joint_targets의50 native steps 뒤 기존 `_forward_physics` 반환 이후 post-qpos/qvel/foot/thorax frame이다. Anchors100/164/228/292와 각각101/165/229/293, 합8개 boundary의 full qpos/qvel/mocap_pos/mocap_quat/physics time/control/network clock/valid를 read/copy만 한다. 기존 capture 직후 같은 body를 읽고 extra live mj_forward/mj_step/neuralreadout0, instance 교체0이다. Last-native cache와 혼용하지 않고 기존 forward를 제거하거나 pre/post 중 오차가 작은 채널을 사후 선택하지 않는다.

각 boundary의 tarsus5 body origins6 world positions, thorax origin/xmat, 원점+각1mm axis를 저장한다. 같은 post-forward state의 read-only mj_jacBody는 별도 결과 버퍼만 쓰고 네 anchor의 full-nv translational J와 active42 hinge dof/qpos IDs를 저장한다. 실제 nq/nv/nmocap 등은 compiled model에서 얻고 bounded shape/dtype를 로딩 전 검사한다. 기존 frontend42/LEGS와 ID/name order를 결합한다. Tip origin은 pad 접촉점이 아니다.

초기화·restore·물리 옵션 적용 후 정상 loop 전에 compiled MJB를 정확히 한 번 저장한다. Local MuJoCo save/load API를 Source/stub로 확인하고 generated synthetic roundtrip으로 검증한다. MJB buffer/temp/durable bytes와 serialization 복제를 host/output cap에 포함하고 Publisher/Source/ownership fence를 우회하지 않는다. MJB bytes/SHA, MuJoCo version, nq/nv/nu/nbody/nmocap/integrator/timestep, active joint/qpos/dof/actuator IDs+names, tip/thorax IDs+names, mm/rad/frame/mutable model settings를 저장한다.

Actuator gain/bias 등의 dynamic mechanical fields가 변할 수 있으므로 전체 model immutable이라고 단정하지 않는다. FK에 필요한 body_pos/quat, joint axes/positions/types/addresses, hierarchy, mocap binding과 ctrl clamp/range fields를 fixed bytes로 검사한다. 변화하는 dynamic fields는 따로 기록/제한하고 geometry fields가 변하면 reconstruction invalid다. MJB exact identity와 geometry invariance를 구분한다. Wrapper는 모든 예외에서 inherited/instance method ownership까지 복원한다. 새 neutral common command/tendon/native/lean trace는 기존 actual tracking neutral과 exact/drift0로 비교한다. Full pose/model은 first-observed다. Partial valid bitmap/receipt/first_error/cleanup failure/비용을 보존한다.

## 별도 CPU 복원과 네 전이

별도 model/data에서 full qpos/qvel/mocap/time을 복원해 mj_forward를 호출한다. Live data/새 FlyGymBody를 분석에 사용하지 않는다. Saved foot world positions/thorax 원점+1mm axes는 abs1e-9mm/rtol0, saved anchor native J는 abs1e-8mm/rad/rtol0로 비교한다. 저장 quaternion을 사후 normalization으로 교정하지 않는다. 필요한 데이터 누락/model mismatch는 invalid다.

Centered FD active hinges±1e-6rad에서 perturbation은 qpos address, J 열은 dof address다. Root/passive/mocap/model/start frame은 고정하고 FD free kinematics에 ctrl clamp를 적용하지 않는다. Generated 작은 사례는 nonzero root/passive/mocap와 qpos≠dof, order permutation, invalid model/shape/dtype/identity, partial capture/wrapper restoration, 실제 synthetic model save/load roundtrip을 확인한다. Implementation을 그대로 다시 계산하는 fixture는 피한다.

네 c→c+1 transitions만 분석한다. Start thorax frame으로 world vectors를 투영해 J_A(q_c)*(effective_ctrl_(c+1)-q_A,c), J_A(q_c)*(q_A,c+1-q_A,c), actual world tip delta를 구분한다. Clamp는 compiled actuator flags/ranges이며 joint range와 다르다. Native interpolation과 final endpoint ctrl를 구분하고 최종 target만으로 전체 interval 물리 결과를 예측했다고 주장하지 않는다.

각 시점 자기 thorax frame의 body-relative tip AP 차이는 별도 열이다. Root rigid motion이 body-relative AP에서 상쇄되므로 그 residual을 root cause라고 명명하지 않는다. World start-frame residual은 root/passive/nonlinear/contact/servo가 혼합된 성분으로 제한한다. 동일 두 boundary load masks로 loaded-loaded/unloaded-unloaded/transition을 구분하고 load threshold/분모는 기존 tracking metadata와 같게 명시한다. 빈 category는 UNOBSERVED,4local anchors를200control stance/full stride로 확대하지 않는다. 힘·추진·미래 접촉 인과 증거가 아니다.

Geometry pass이면 ROOT는 RH association이 command direction/actual active change/혼합 residual 중 어느 관측과 연결되는지 작은 표로 판단하고 다음 contact-conditioned neural objective를 설계한다. 실패면 최초 차이와 비용을 보존하고 시점/복원/구현 문제를 먼저 조사한다. Threshold 완화/자동 동일실험 반복0이며 이 stage에서 후보 채택·자연 보행 완성은 없다.

## 소스 소유권과 보존

기존 observer/runner를 재사용하고 필요한 observer factory/small hook만 추가한다. Tracking c archive8과 live replay freeze a는 보존한다. Shared owner 변경 전 실제 live replay에 사용한6개 canonical Source를 bytes/SHA로 archive하여 역사 Source와 current Source를 분리한다. 현재200debits, offset 뒤15 Original과 실패/새3generated qualification의 실제 비용을 이어 새로운 역할/Source/qualification을 묶는다.

Source-only Sol은 proposals/handoff만 쓰고 canonical/tests/Original/ledger/grants/import/AST/numericalarrays/추가 subagents0이다. ROOT가 검토 후 적용과 실제 실행/비용을 수행한다. 활성 plan/progress에 결과와 판단을 append한다. Root GOAL_PLAN.md151552B/SHA16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace는 불변이다. F1–F8/M2/UI/provenance/unused F6/기각 이력/biology UNPROVEN을 유지하고 Source/검증 완료를 보행 개선으로 표시하지 않는다.
