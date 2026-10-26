# E229 실제 접촉점 분해 결과

사건25의 오른쪽 앞다리(RF)가 자기 관절 운동으로 발을 앞으로 미끄러뜨린다는 가설은 **기각**한다. 원래 명령으로 재현한0.2초의 시작·종료 snapshot,41물리행,2000native 접촉·반력 표본은 E225/E227/E228 sham과 정확히 일치했다. 관측 추가가 신체 움직임을 바꾼 결과가 아니다. 자연 보행을 개선한 새 후보는 아직 없다.

고정 `during_break`(control1623–1630)에서 RF의 지지 제동305표본을 실제 접촉점으로 분해하면 하중 가중 전방 속도는−0.098488mm/s였다. root6자유도의+0.875686mm/s와 자기7관절의−0.975843mm/s가 거의 상쇄했고, 나머지는+0.001669mm/s였다. 접선 미끄럼 크기는0.419635mm/s다. 뒤쪽 반력이 크다는 사실을 발을 앞으로 미는 관절 동작과 동일시할 수 없다.

| RF 관절 | 지지 제동 접촉점 AP 기여(mm/s) |
|---|---:|
| coxa yaw | +0.285975 |
| coxa roll | −0.733537 |
| coxa pitch | −0.876986 |
| femur roll | +0.199935 |
| femur pitch | +1.166549 |
| tibia pitch | −0.925666 |
| tarsus pitch | −0.092113 |

여러 관절의 양·음 기여가 상쇄한다. 한 관절의 양의 기여만 제거하면 좋아진다는 인과 결론도 아직 없다. 같은 구간 RH는 짧은 지지 제동25표본에서 자기관절 AP가+15.955226mm/s였지만, 전체 지지234표본의 순 반력은+0.006439BW·s로 추진이었다. 다리 역할은 구간과 부분집합에 따라 달라진다.

이번 support는48개 leg segment의 **양의 normal load 합≥.02BW**다. 기존 E228의 tarsus5 normal load 지지와 같은 마스크라고 가정하지 않는다. signed AP와 미끄럼 크기, 순간 접선 힘×속도와 생물학적 근육 일도 구분한다. post-integration shadow 반력은 직전 solver impulse가 아니며, 저장 COM/force 비교에 남은 pre/post 편향도 해결됐다고 주장하지 않는다.

실제 gpt-6-sol/high가 구현하고 부모 FIRST에서 결함0, 신규 합성5시험/Ruff/컴파일을 확인했다. 별도 부모 원시 수식 검산을 준비한 뒤 단일 원본을 실행했다. 모든 활성 fly-floor 접촉5000개 중 양의 leg contact4863개에9726회 Jacobian을 계산했다. 비양수137개도 raw에 보존했으며 비다리 접촉0이다. 최대 분해 오차1.11e−14, 전체 leg 힘 합 오차2.22e−16BW, full-fly 힘 합 오차8.88e−16BW였다. 부모의 독립 raw·집계·파일 검산839항목이 통과했다. 검산 개수는 실험 반복 수가 아니다.

원본 감독10.6917021초, 내부7.1977479초(구성3.6477852초), 과학2000steps/0.2초, body1/shadow1, neural/CUDA/fit0/retry0이다. nominal settling2000은 소스 유래로 별도이며 직접 계측하지 않았다. 최대 process1,264,308,224B/tree1,269,276,672B, 출력10,353,466B, 소유 잔류 프로세스0이다. 원본 보수누적**6051.5664745996655/6084초**, 잔여32.433525400334474초다. 부모 raw 검산.798333초/명령1.2824584초와 생성 시험·읽기 비용은 별도로 기록했다. 부모 생성 fixture의 초기 NumPy bool JSON 게시 오류도 원본 실패와 구분해 보존했다.

다음은 저장 root6자유도·접촉 segment·지지 전환을 검토해, 실제 지지력 전달을 바꾸는 한정된 인과 개입을 설계하는 것이다. 전체 다리 hold나 일괄 gain 탐색을 자동 반복하지 않는다. E225 startup 실패·10초 미완료·적격 완전 주기0, E228 두 개입 기각을 유지하며, F1–F8·전체 finite-goal은 active/미완료다. E150 최선/M2/UI/F6 미노출을 보존했다.

[사전 설계](BANC_M3_E229_EFFECTOR_CONTACT_DECOMPOSITION_DESIGN_20260925.md) · [독립 검산](../verification/banc-natural-gait-m3-20260919/parent-e229-output-review.json) · [원자료](../verification/banc-natural-gait-m3-20260919/e229-effector-contact-decomposition/result.json) · [감독](../verification/banc-natural-gait-m3-20260919/parent-e229-supervisor.json)
