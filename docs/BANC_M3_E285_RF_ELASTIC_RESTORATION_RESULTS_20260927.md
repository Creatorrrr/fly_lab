# E285 결과 — 실제 오차 감소는 확인, RF 최종 시작 여유는 미달

2026-09-27. 실행 유효, 최종 가설은 bounded6회에서 기각이다. 신경·decoder·source family를 유지한 새 elastic 보정은 실제 q peak를 .03942350046875959에서 .03693922689093462 rad로 줄였다. 최종 한도 .0359964에는 못 미쳤으며 여섯 outer 후 정해진 대로 종료했다. f32·전체 초기화·full-fit·BANC/body·F6·제품 채택은0이다. 이 결과는 자연 보행 성공이 아니다.

모든6 QP의 원래 P/A/b/dual 인증과 실제 수용 결정을 root가 저장 자료에서 독립 검산했다. 실제 peak 순서는 .036993858999 → .036965882753 → .036949550901 → .036942100540 → .036939235049 → .036939226891이다. 둘째 outer의 alpha1은 실제 .038443297294로 악화돼 거절하고 alpha.5만 수용했다. 나머지는 alpha1을 수용했다. 마지막 개선량은8.157584989e-9 rad다. 모든 중간 실제 q SSE/LTM/계수 기준은 통과했으며 최종 q SSE .25935787833740015다. 최종 peak만 미달이다.

추가 peak-only 저장 dual은 local step .5/norm/LTM/solver reserve를 제거하고 원래 hard coefficient 영역을 써도 모든6 fixed J에서 최종 목표 불가능성을 지지한다. 마지막 fresh J의 하한 .036936296347271205는 실제 정체값과 가깝다. 주된 dual 충돌은 RF 첫 관절 control99의 음의 오차와 control83의 양의 오차다. 시간·반복 횟수만 늘려 해결된다고 판단하지 않는다. 이 인증도 국소 J에 한정되므로 비선형 family의 전체 불가능성을 주장하지 않는다.

Actual gpt-6-sol/xhigh 구현, root FIRST에서 수정 요구0. Agent 생성 검사 중16개 실패는 mock deepcopy/종료시점 fixture 오류였고 수정 뒤88개를 검증했다. Root는 최종 전체88PASS/Ruff와 별도 선형·비선형·flat 응답의 수학 검산을 통과했다. Agent tiny14회/218iter, root12회/195iter, 원본과 별도 회계다. Helper SHA871961f49d0840236251fa54ba91dd182536c6f8639447334db3127f8cc3dd1b와 runner9ce420c9800a7fcd9d7d8597c532460504d0705697e8aface67b3979a3941506을 동결했다.

원본13 engine/평가, forward26000/derivative12000ticks, QP6/162iter. Constructor .105871100037초/solve2.676623899955초/certificate.034103599959초, supervisor5.947030299983453초/RSS475258880bytes. Owned PID32224·41932 종료. 원본 누적7861.705837599529/7905초, 잔여43.29416240047067초. E280 자원 보장 철회 및 모든 이전 기각/F1–F8/M2/E150/UI/미사용 F6를 보존한다.

직접 근거는 `verification/banc-natural-gait-m3-20260919/parent-e285-original-final-review.json`, `parent-e285-plateau-audit.json`, `parent-e285-output-manifest.json`, `parent-e285-final-source-review.json`, 동결32source 및 원본 raw다. 다음 진단은 RF control99의 실제 운동뉴런·필터·근육 출력을 보존해 신경 발화와 decoder 분기 중 어떤 제한이 지배적인지 구분한다. 아직 계산하지 않은 내부 상태를 추정값으로 대체하지 않는다.
