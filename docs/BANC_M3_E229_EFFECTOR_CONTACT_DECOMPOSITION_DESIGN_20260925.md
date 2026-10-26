# E229: 현재 effector의 실제 접촉점 속도와 관절 기여

E228의 RF/RH 전체 다리 명령 유지는 둘 다 고정 개선 기준을 통과하지 못했다. RF 제동 반력 순위만으로 RF 전체 출력을 낮추는 것은 근거가 부족하다. E227 저장 질량중심 가속도 검산은 단절 중 반력과 높은 동시 상관(.96935)을 보였으나 pre/post 평균 편향과 잔차가 남았다. 따라서 힘 합을 완전한 운동량 수지나 특정 근육의 원인으로 사용하지 않는다. E157의 접촉 Jacobian 방법은 E150/RM에서 이미 검증했지만 현재 effector/RF에 같은 결론을 적용할 수 없다.

이번 단계는 **진단**이다. 가설A는 단절 중 RF의 지지 접촉점이 자기 관절 움직임으로 전방 미끄럼을 만들고 있다는 것이다. 경쟁 설명은 root 움직임, 다른 자유도, 접촉 전환 또는 다른 다리의 힘 상쇄이다. 접촉점의 속도 기여를 직접 측정해 후속 관절/신경 전달 개입을 좁힌다. 기존 전체 다리 hold를 반복하거나 새 gain을 탐색하지 않는다.

## 고정 재현과 관측

- E227 `event-start-25`의 control1607/8.035초 body-only 상태에서 E225 원본 control1608–1647의40명령만 그대로 진행한다. 신체 생성1, 0.2초/2000native steps, 신경·CUDA·fit0, 원본1/retry0이다. E228 sham과 같은 원본42목표/adhesion6개off/힘줄8개/interpolate=True 경로다.
- 시작·종료 snapshot은 E228 sham과 정확히 같아야 하고, 모든41물리행, native6필드와 force11필드도 기존 E225/E227/E228 sham과 dtype/shape/bytes까지 같아야 한다. 재현 차이는 기술 실패로 보존하고 그 자료로 기전 결론을 내리지 않는다.
- E227과 같은 post-integration shadow `mj_forward` 상태에서 접촉을 관측한다. 모든 활성 fly-floor 접촉의 geom ID, contact index, 실제 접촉점, frame, wrench, fly 방향 sign, leg/link, normal load를 기록한다. 여섯 다리의 48 segment mapping을 사용하며 tarsus5 원점으로 실제 접촉점을 대체하지 않는다. 비다리·0이하 하중 접촉도 원자료와 제외 개수/힘으로 남긴다.
- 양의 normal force를 갖는 모든 leg-floor 접촉에 대해 실제 접촉점에서 fly segment와 지면 body의 `mj_jac`를 각각 한 번 계산한다. native qpos/qvel/clock, 두3×nv Jacobian과 요청/반환 여부를 저장한다. 관측 실패는 완료한 물리 clock·native 상태·원시 접촉·이미 반환된 Jacobian까지 보존한다. jacobian 두 개가 반환되지 않은 행은 유효 분해로 집계하지 않는다.
- `tools/analyze_c_banc_m3_native_contact_decomposition.py`의 검증된 `validate_layout` 및 `decompose_contact` 수식을 재사용한다. 다리7 DOF와 root6 DOF는 모델에서 검사한다. 초기 trace0의 고정 native XY 방향으로 계산한다. 현재 회전에 따른 전방 투영 변화는 개입하지 않는다.

## 수식·집계·판정

접촉 frame의 법선 n에 대해 P=I−nnᵀ, J=P(Jfly−Jfloor), v=J qvel이다. 전방 성분을 root6, 해당 다리7, 나머지 DOF로 분해한다. 접선 힘은 원래 contact wrench의 world 변환에서 법선 성분을 제거한다. 접선 힘×속도도 같은 DOF로 분해한다. 이는 순간 접촉 기여이며 근육 자체의 힘·에너지나 직전 solver impulse가 아니다. 회전 접촉 토크의 일은 포함하지 않는다.

E228과 같은 `lead`1608–1619, `hold_window`1620–1635, `recovery`1636–1647, `during_break`1623–1630을 쓴다. 여기의 hold_window는 원본 재현의 시간 구분명으로, 명령 hold를 적용하지 않는다. 각 native 표본의 다리 양의 normal load 합이≥.02BW이면 지지로 분류한다. 다리별/phase별로 전체 지지와 순 초기방향 힘<0인 지지 제동 부분집합을 모두 보고한다.

모든 다리의 normal-weighted 접촉점 AP 속도·slip 크기·root/own7/residual 기여와 각7관절 기여, 유효 접촉수/지지 시간/하중합, signed 전방 힘 시간합 및 접선 힘×속도 시간합을 저장한다. 속도에는 하중을 가중하되 이미 힘이 포함된 power에는 하중을 다시 곱하지 않는다. 다중 접촉의 힘·power는 각 native 단계에서 합산하고 dt=.0001을 곱한다. 지지 시간은 접촉 개수가 아니라 고유 native 표본 개수로 계산한다. 미지지/무자료 평균은0이 아니라 null이다.

재현 exact, 모든 해당 접촉 포함, 물리 건강, 분해 closure/direct/projection 오차≤1e−10이면 진단이 유효하다. 관측한 leg 접촉 힘의 합이 E227 all_leg force와 허용오차1e−10 내 일치해야 한다. 가설A는 고정 during_break RF 지지 제동 부분집합에서 접촉점 AP와 own7 AP가 모두 양수일 때 그 국소 운동학적 설명을 지지한다. own7 AP≤0이면 이 설명을 기각하고, 부분집합이 없거나 관측 불완전이면 불충분으로 남긴다. root/residual 및7관절 순위도 함께 보고하며 특정 관절을 바꾸면 개선된다는 결론은 별도 물리 개입 전에는 내리지 않는다. 한 사건의 결과를 열 사건이나 최종 자연 보행으로 일반화하지 않는다.

## 구현·예산·보존

실제 gpt-6-sol/high가 `tools/observe_c_banc_m3_effector_contact_decomposition.py`와 `tests/test_c_banc_effector_contact_decomposition.py`를 구현한다. 부모 FIRST→필요시 같은Sol 첫수정→SECOND→부모 잔여수정 순서다. 합성/mock 검사 후 부모 승인 소스와 freeze 해시로 한 번 실행한다. 부모는 raw Jacobian에서 직접 분해·집계를 검산하며 구현 집계 함수를 답으로 사용하지 않는다.

CPU1/GPU0, body1/shadow1, 과학2000steps/40controls, 최대16000 raw 접촉/32000 Jacobian calls, fit/solver/neural0. nominal 구성 settling2000은 소스 유래이며 계측값으로 부르지 않는다. 내부30초(마감5초 예약)/외부35초/감독40초, RSS6GiB/출력128MiB/free2GiB. 현재 원본 보수누적6040.874772499668/배정6084초, 잔여43.125227500332초에서40초를 예약한다. 배정 증액0/자동 재시도0이다. E157의1초 전체 접촉 관측43.31초와 E228의0.6초 반력 관측11.00초를 비용 근거로 사용하며, 한도에 걸리면 부분 원자료와 실제 비용을 보존한다.

F1–F8/기존 evaluator를 바꾸지 않는다. E225 startup qRMS/max 실패, 10초 미완료, 완전 적격 주기0, E228 두 hold 기각, E150 최선 검증 산출물/M2/UI/F6 미노출을 보존한다. 전체 finite-goal은 active·미완료다. 이번 진단은 후보 채택이나 F2 달성 근거가 아니다. 가설A가 지지되면 지지 제동과 정상 추진에서 기여가 어떻게 다른지 검토하여 한정된 관절 경로 인과 실험을 별도 설계한다. 기각되면 전체 다리 약화 대신 root/수동 접촉/다리 간 전환 설명으로 방향을 수정한다.
