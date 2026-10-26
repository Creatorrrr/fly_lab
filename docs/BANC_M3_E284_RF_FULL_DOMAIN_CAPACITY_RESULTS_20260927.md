# E284 결과 — RF 중립의 직접 선형 보정은 원래 계수 영역에서도 기각

2026-09-27. 실행은 유효하며 RF capacity 가설은 이 방법으로 지지되지 않았다. 중립 q/LTM/두 Jacobian을 정확히 재현한 뒤 첫 QP가 `PrimalInfeasible`로 끝났다. 보정 실제 평가·f32·전체 초기화·full-fit·전체 BANC·신체·F6·후보 채택은 모두 0이다. M3 F1–F8은 미완료다.

Root는 원자료에서 hard-domain P/A/b, RF 축, dual cone과 계수 box 상한을 독립 재구성했다. 전체 문제의 weighted-slack 상한은 -0.9999440983402713이다. LTM 제약을 제거해도 같은 인증의 상한 -62.83811740062635가 불가능성을 지지한다. q SOC의 dual을 단순 삭제했을 때의 양수는 불충분한 인증이며 q norm이 원인이라는 뜻이 아니다.

별도 저장 대수 검산에서는 RF peak가 이미 q SOC를 함의함을 이용해 SOC dual을 부호별 peak dual로 옮겼다. LTM과 norm 제약 없이, solver reserve 1e-6도 제거한 실제 peak 조건에서 상한 -64.24975404712724, 중립 고정 J의 peak 하한 .03601578809214704를 얻었다. 최종 .0359964보다 크다. 작은 neutral trust나 norm quota만 제거해서 해결된다는 설명은 기각된다. 이 결론은 고정 J에 관한 것이며 전체 비선형 해의 불가능성은 아니다.

RF moment 행렬 rank는 7/7이고 자유 activation의 목표 투영 잔차는 1.67e-16 이하였다. 이것은 양의 생리적 activation이나 신경망의 도달 가능성 증거가 아니다. 특히 RF 첫 관절의 중립 J에서 비영 응답을 갖는 것은 16개 edge, MN 593/74561/141493 세 개다. E283 중간 실제 점의 해당 J는 최대 .1031008622 변했고 동일 full-domain peak 인증은 양수여서 불충분하다. 다음 설계는 불가능한 최종 선형 문제를 매번 요구하는 방식 대신, 실제 peak 감소를 확인하며 fresh J를 갱신하는 유한 비선형 restoration을 비교한다. 최종 실제 기준은 유지한다.

Actual gpt-6-sol/xhigh FIRST72검사/Ruff와 root72검사/Ruff·독립 analytic2문제가 통과했다. Root FIRST 수정 요구0. Agent tiny12회/158iter와 root8회/103iter는 별도 생성 비용이며 원본이 아니다. 최초 Ruff B023 오류 및 수정도 기록했다. 원본 engine1/평가1/forward2000/derivative2000, QP1/14iter, constructor .015081600053초, solve .260797200026초다.

Supervisor2.2092928999918513초/RSS470700032bytes/출력12021172bytes, owned PID17644·45428 종료. 원본 누적7855.758807299546/7905초, 잔여49.24119270045412초. E280 실제 엔진 생존 보장 철회, E277/E279/E281–E283 기각, M2/E150/UI/미사용 F6를 보존한다.

근거: `verification/banc-natural-gait-m3-20260919/parent-e284-original-final-review.json`, `parent-e284-peak-structure-audit.json`, `parent-e284-output-manifest.json`, `parent-e284-final-source-review.json`, 원본 freeze·source-review·supervisor와 raw output. 현재 실행/예약0이다.
