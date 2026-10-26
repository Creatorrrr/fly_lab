# E283 두 단계 시작 보정 결과

2026-09-27. [사전 설계](BANC_M3_E283_STARTUP_CONTINUATION_DESIGN_20260927.md)에 따라 원본을 한 번 실행했다. 중립 여섯 다리의 q/LTM/J는 exact 재현했고, 중간 단계의 두 QP는 모두 `Solved`와 독립 primal/dual 인증을 통과했다. 실제 CUDA 응답은 중간 기준을 만족하지 못했다. 판정은 **유효 진단·초기화 기각**이며 최종 단계와 f32 확인은 실행하지 않았다.

중간 한도는 peak .0379962, 전체 q SSE 1.215597624002408이다. 두 번째 실제 보정의 SSE는 1.2138731817913084로 통과했지만 RM peak .03799725414270316, RH peak .03799635661631784가 각각 1.054142703162e-6, 1.566163178382e-7 초과했다. RM control85/99 local joint4, RH control69 local joint5의 세 행이다. 다른 네 다리와 LTM/계수 조건은 통과했다. 이를 반올림 허용오차로 흡수하거나 세 번째 보정을 추가하지 않았다.

첫 보정의 최대 실제-선형 q 차이는 .0115326081364, 두 번째는 .00401984417428이었다. 최대 차이가 최종 peak 행과 일치하는 것은 아니다. 두 번째 RM peak의 실제-예측 차이는 약 2.11e-6으로 선형 여유 1.0556e-6보다 크다. Root는 원래 실제 q/J에 18/19가 함께 적용됐는지, 전체 sparse P/A/b, 원래 중립 trust, 안쪽 f32 구간, 두 원시 해와 실제 평가 계수의 일치, objective/dual/gap/상보성을 저장 원자료로 독립 확인했다.

더 중요한 제한은 중간 단계의 작은 초과만 해결해도 최종 문제가 해소되지 않는다는 점이다. E282의 RF nonnegative dual을 새로 저장된 실제 J에 적용했다. 중립의 RF 선형 peak 하한 .0368050031208은 첫 중간 보정 후 .0367589028082, 두 번째 후 .0367589028606으로 남았다. 모두 최종 .0359964보다 크다. q norm/LTM/다른 다리 없이도 성립하며, 고정된 각 선형 근사에만 해당한다.

동일 저장 dual과 원래 f64 box를 사용한 추가 대수적 계산에서 탐색 반경 .05를 .1로 넓혀도 상한은 중립 -8058.16172834, 마지막 중간점 -7032.09114209로 여전히 음수였다. 따라서 작은 반경 확대만 시험할 근거도 약하다. `.5570713707`와 `.3515234086`은 두 점 사이 affine 기울기를 외삽한 witness의 영점 추정이며, 그 반경의 실행 가능성이나 유효 인증이 아니다. 이 계산에 새 solver/model 호출은 없다.

다음 질문은 원래 source 계수의 hard domain `[0,8]` 및 triple sum interior 안에서도 RF의 최종 peak를 만족할 수 없는지, 아니면 중립 근처의 탐색 제한이 주된 원인인지다. 기존 신경 노드·연결·세 시간 성분을 유지하는 RF만의 bounded capacity 진단을 별도로 설계한다. 기존 실패를 통과로 바꾸거나 최종 startup/F1–F8을 완화하지 않는다.

Actual gpt-6-sol/high가 두 신규 파일을 구현했다. 생성 테스트33/Ruff와 root의 독립 affine 2개 QP 검산을 통과했고 FIRST에서 수정 요구는 없었다. Agent 실제 tiny solver 누적10회/100iter, 생성 검사 wall10.3232856초(초기 fixture 오류2회 포함), root tiny4회/46iter·supervisor4.1056508초다.

원본은 engine18/실제평가18, QP2/47iter, forward/derivative 각각36,000ticks, supervisor8.497852200002초, RSS690,843,648bytes, 출력220,214,992bytes였다. Owned PID35384/39664 종료를 확인했다. 누적7853.549514399554/7905초, 잔여51.450485600446초. Root 원자료 검산 .802259299962초, 저장 RF bound .131698200013초, 반경 감도 계산 .088159800041초는 모델/solver0이다.

전체 BANC/body/physics/train/development/fullfit/F6/후보 채택은0이다. 자연 보행 목표 F1–F8은 미완료다. 증거는 `verification/banc-natural-gait-m3-20260919/parent-e283-original-final-review.json`, `parent-e283-rf-saved-linear-bound.json`, `parent-e283-radius-sensitivity.json`, `parent-e283-output-manifest.json`, `e283-startup-continuation/`에 보존한다.
