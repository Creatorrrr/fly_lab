# E120 제안: 남은 개선 방향과 초기 제약의 충돌을 구분

2026-09-20 · 부모 설계안 · 아직 실행/입력/자원 동결 및 구현 전

E119는 모든 초기 sample을 제약하는 새 적합을 한 번 실행했다. 여섯 다리가 실제 초기 제약에 맞는 선택점을 얻었고 CUDA/NumPy 및 원자료 검산도 통과했지만, 기존 발 위치·속도20%개선 기준은 실패했다. 여섯 다리 모두100 unique 평가 한도에 도달했으며 수렴을 증명하지 않았다.

E115 시작점 대비 선택점의 true worst objective 상대 감소는 LF .000113157, LM .000002076, LH .000026406, RF .000001520, RM .000018039, RH .000007325다. 일부 delta는 .04~.08 변했으므로 단순히 모든 변수가 거의 움직이지 않았다고 설명할 수 없다. 선택점 loss gradient L2도 .33~2.86으로0이 아니다. 이 값은 저장 raw의 기술 통계이며 constrained stationarity나 전역 표현 용량을 증명하지 않는다. 근거는 e119-samplewise-fit/parent-execution-provenance.json이다.

다음 질문은 **동일한 허용 가중치 범위와 초기 자세 기준 안에서 유효한 국소 개선 방향이 남아 있는가**다. 먼저 E119의 선택 eval에 이미 저장된 두 loss gradient와 모든 signed startup Jacobian을 사용한다. forward/derivative bank를 다시 실행할 필요가 없다.

검토할 방법은 h의 작은 고정 범위와 원래 delta bounds 교집합에서 두 문맥 loss의 affine epigraph 감소를 구하는 것이다. 기존 E116의 현재 최대 제곱오차 gradient를 재사용하지 않는다. q+Jq*h 및 LTM+JLTM*h의 모든 signed sample을 제약하고, RMS는 이 선형화된 residual의 실제 제곱평균으로 검사한다. 기존 SELECTOR_CAP 안의 feasible set을 연구 대상으로 명시하고 실제 후보의 원래 채택 기준은 바꾸지 않는다.

비교는 같은 입력·같은 h bounds의 startup 제약 없는 선형 문제와, 모든 signed sample/RMS 제약을 넣은 국소 볼록 문제다. 기존 HiGHS 선형 solver를 재사용하려면 RMS에 대한 supporting cuts를 제한된 횟수 추가하는 outer approximation을 고려한다. LP relaxation의 lower bound와 실제 local RMS-feasible 점의 upper bound를 구분하며, primal/dual residual과 bound gap을 독립 검산한다. cut 한도까지 feasible 점 또는 작은 gap을 확보하지 못하면 미해결로 기록하고 국소 stationarity를 주장하지 않는다. h=0의 실제 선택 기준 통과 여부도 raw에서 재확인한다.

이 비교에서 충분한 feasible 감소 방향이 확인되면 그 방향에 대한 소수의 실제 비선형 acceptance를 사전 동결하여 더 직접적인 trust-step 적합으로 진행한다. 제약을 넣을 때만 감소 여지가 작아진다면 초기 안정화와 지속 운동을 같은 정적 가중치로 함께 맞추는 구조의 충돌을 분석한다. 그 경우에만 BANC 내부의 시간적 신호 분리·적응 상태 등 새로운 생리 가설을 설계한다. 현재의 한정 실패만으로 그런 변경이 필수라거나 BANC의 보행 능력이 없다고 단정하지 않는다.

E115/E119의 반복 수·범위·성공 기준을 늘리지 않는다. E120의 정확한 solver 호출 수, cut 상한, 수치 허용치, wall budget, 입력/source SHA는 구현 전에 별도 설계와 freeze로 고정한다. 이 제안만으로 LP/fit/모델 응답을 실행하지 않는다. 외부 CPG/위상/궤적/root 교정 금지, 전체 F1–F8, 마지막 실제 E113 F3/F4/F5 실패, M2 원본 및 F6 미노출을 유지한다. 개발과 리뷰 순서는 부모 설계→Sol 구현→부모 첫 리뷰→같은 Sol 첫 수정→부모 재리뷰/잔여 직접 수정이다.
