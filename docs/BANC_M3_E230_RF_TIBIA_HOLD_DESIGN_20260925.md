# E230: RF 경골 관절 하나의 국소 신체 인과 비교

E229의 정확 재현은 RF의 단절 지지 제동에서 자기7관절 합 AP−.975843mm/s, root+.875686mm/s를 확인했다. 자기관절이 발을 앞으로 미는 가설은 기각됐다. 그중 tibia-pitch의−.925666mm/s는 음의 자기관절 기여 중 가장 컸으며 femur-pitch의+1.166549mm/s와 상쇄했다. RF 실제 접촉은 tarsus3였다. root 두 번째 회전 DOF는+2.167169mm/s, X 병진은−.710702mm/s였다. 이는 속도 분해이며 특정 관절이 제동 원인이라는 증거는 아니다.

E228의 RF7관절 hold는 반력을10.59% 늘리고 후진을45.74% 늘렸다. 여러 자유도를 동시에 바꾼 결과로 단일 경골 관절 효과를 알 수 없다. 실제 gpt-6-sol/medium의 읽기 전용 검토와 부모 검토는 **한 관절 개입**을 다음 판별로 선택했다. E85 외부 참조도 모든 tendon .2를 사용하므로 현재LTM 포화 자체를 원인으로 지목하거나 임의로 해제하지 않는다. 과거 M2 활성 길이/강성 및 활성 감쇠 실패도 보존하며 반복하지 않는다.

## 가설과 고정 비교

질문은 동일한 다른 명령을 유지할 때 RF tibia-pitch의 국소 명령 변화가 후진과 제동력을 수정할 수 있는 경로인가이다. 경쟁 설명은 나머지 관절·root 회전·다리 간 힘 상쇄가 우세하여 이 한 관절을 유지해도 개선되지 않거나 필요한 지지가 손상된다는 것이다. 이 실험은 진단이며 runtime 시간 기반 hold를 최종 신경 제어로 채택하지 않는다.

1. E227 사건25 body snapshot(control1607/8.035초), E224의 같은 world/seed/parameters에서 body를 한 번 구성한다. 각 arm은 같은 snapshot으로 복원한다. 순서는 `sham`, `rf_tibia_hold`이며 각 control1608–1647/40명령/2000native steps다. 신경/CUDA/fit0이다.
2. sham은 E228 sham 및 E229의 시작·끝 snapshot/41물리행/native6/force11/actual-contact/Jacobian 원자료와 정확히 같아야 한다. sham이 완전하고 건강하게 재현되어야 다음 arm을 진행한다.
3. `rf_tibia_hold`는 control1620–1635에서 **관절26 하나만** 원본 control1619의 목표값으로 유지한다. control1636부터 원래 목표로 복귀한다. 나머지41목표, 복부/6LTM 힘줄, adhesion6개off, physical 모델과 모든 신경 상태는 그대로다. 5ms/50step/interpolate=True 경로를 보존한다. 크기·부호·창의 추가 탐색은 없다.
4. 두 arm의 control1608–1619까지 명령/물리/native/force/contact/Jacobian을 정확 대조한다. 실제/요청 명령과 최초 qpos/ctrl/native 차이를 보존한다. source global index50950 이전 차이는 비교 실패다. 제어 경계 중심 차분이 한 행 앞에 차이를 만들 수 있으므로 실제 물리 차이로 오인하지 않는다.
5. E229 관측을 각 arm에서 사용한다. 모든 활성 floor-fly raw contact, positive leg contact의 실제 접촉점 Jacobian 두 개, native 상태, 접촉/반력, root6/own7/잔여 AP와 접선 power를 보존한다. 건강 중단/관측 실패 시 이미 완료한 clock·상태·부분 contact/Jac 요청·반환과 오류를 보존하고 재시도하지 않는다. 기술·자원 실패 뒤 다음 arm을 진행하지 않는다.

## 측정과 결정

E228의 고정41행 속도 계산과 control1620–1646 요약을 그대로 쓴다. 후진거리, 최소 초기방향 속도, 전체1607→1647 전진변위, yaw, upright/nonfoot,42관절 추종과다리별 slip을 보고한다. 추가로 force11의 실제 질량중심을 초기 native XY 방향에 투영한 위치·속도/단절 중 후진거리와 native root angular 속도를 저장자료에서 계산한다. 기존 thorax 기준을 COM 기준으로 바꾸어 통과시키지 않는다.

phase는 E229의 lead1608–1619, hold_window1620–1635, recovery1636–1647, during_break1623–1630이다. E229 동일 positive all-leg normal sum≥.02BW 지지·초기방향 음의 순 반력에서 AP/root/own7/잔여, 실제 contact segment, unique-native 지지시간 및 signed 힘/접선 power를 비교한다. 다른 다리의 보상도 모두 보고한다. E228의 tarsus5 지지 힘 기준도 별도 계산하여 기존20% RF 반력 조건과 직접 비교한다. 두 support 정의를 섞지 않는다.

원래 E228 **국소 개선** 기준을 유지한다: 완전·건강, sham 대비 후진거리≥25%감소, 최소 전방속도≥.5mm/s증가, 전체전진손실≤.02mm. 여기에 E228 same-support during_break RF 음의 반력≥20%감소를 함께 만족하면 이 단일 경골 명령 경로의 국소 기계적 기여를 지지한다. AP 분해 또는 RF 반력만 바뀌고 실제 후진이 개선되지 않으면 개선 후보로 기각한다. 전체 미끄럼/타다리 force 악화는 별도 기록하고 최종 후보 채택과 구분한다. 완전 창을 못 마친 arm에 전체 개선 점수를 붙이지 않는다.

이 국소 성공도 F2나 F3–F8, 열 사건의 일반화, 생물학적 기전을 입증하지 않는다. 성공하면 시간 hold를 제거한 기존 해부학적 tibia 근육 경로의 인과적 출력 수정과 startup 보호를 별도 설계한다. 실패하면 이 한 관절 hold를 기각하고 추가 anchor/기간/관절 수치 검색으로 이어가지 않는다.

## 구현과 자원

실제 gpt-6-sol/high가 `tools/probe_c_banc_m3_rf_tibia_hold.py`, `tests/test_c_banc_rf_tibia_hold.py`만 구현한다. E228의 검산된 명령·지표·회계와 E229의 실제 접촉 관측/수식을 명시적 인자로 재사용하며 frozen source나 module globals를 수정하지 않는다. 부모 FIRST→필요시 동일Sol 첫수정→SECOND→부모 잔여수정이다. 부모는 관절26 외 명령 불변·sham/lead exact·원시힘/수식·비교 판정을 독립 검산한다.

CPU1/GPU0/원본1/retry0/body1/shadow1, 과학4000steps/80controls/0.4초, 최대32000 raw contacts/64000Jac, neural/fit/학습solver0이다. 구성 nominal settling2000은 직접 계측과 구분한다. 내부30초(마감5초예약)/외부35/감독40초,RSS6GiB/출력128MiB/free2GiB. E229의0.2초Jac 관측10.6917초(구성3.6478초)를 근거로 두 arm의 비용을 제한한다.

현재 누적6051.5664745996655/배정6084초·잔여32.433525400334474초를 보존하고 **원본 실행 전 배정을10초 추가하여6094초**로 기록한다. 새 잔여42.433525400334474초에서40초를 예약한다. 실패한 한도의 소급 확대나 자동 재시도가 아니며, 전체 작업에 사용자가 정하지 않은 시간/토큰 상한을 새로 부과하지 않는다. 준비·검산·서브에이전트 읽기 비용은 별도로 기록하고 미계측을0으로 주장하지 않는다.

E225 startup qRMS/max실패/10초 미완료/완전 적격 주기0, E228 두 hold 기각, E229 운동학 가설기각, E150최선/M2/UI/F6 미노출을 유지한다. F1–F8과 전체 finite-goal은 active/미완료다.
