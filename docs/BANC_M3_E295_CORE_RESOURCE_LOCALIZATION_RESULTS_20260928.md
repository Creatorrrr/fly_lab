# E295 — 핵심 회로의 직접 입력은 같지만 재귀 전달과 발화는 달라졌다

2026-09-28. 저장 자료 분석과 root 독립 검산을 완료했다. 새 신경망·신체·CUDA·solver·fit·원본 실행은0이다. E294의 startup 기각 판정은 그대로다.

원래24core 전체와391MN,96근육,42관절,6힘줄을 control1–100(.005–.5초)의 같은 저장 시점에 대응했다. E150 observed_ids가 정렬되지 않았으므로 명시적 ID map을 사용했다. E150과 E257 core의 rate/drive/adaptation은 정확히 일치했고 근육 활성의 저장 정의도 일치했다.

| 관측량 | E294와 E150의100행 차이 |
| --- | --- |
| Core24 직접 drive / adaptation | 전부 exact 동일 |
| Motor391 직접 drive | 전부 exact 동일 |
| Core24 rate RMS / 최대 차이 | 45.7697342 / 120.7939453 모델 단위 |
| Motor391 rate RMS / 최대 차이 | 14.7260111 / 144.4799185 모델 단위 |
| 42관절 명령 RMS / 최대 차이 | .2295428960 / 1.2941063086 rad |
| 처음 저장된 bit 차이 | core/MN rate, MN adaptation, 근육/관절/힘줄 모두control1(.005초) |

직접 drive가 같다는 것은 해당 저장 필드에 한정된다. 다른 신경을 거치는 간접 감각 피드백이나 모든 recurrent current가 같았다는 뜻이 아니다. 첫 차이는5ms 저장 격자에서 관측된 시점이며 substep 시작 시각이나 인과 선행 관계를 판별하지 않는다.

기존 선택 연결 중24core로 향하는 양수 전달 손실을 같은 candidate rate에서 분해했다. 전체 선택 core-input 손실 중 core 자체가 source인 비중은 .05초56.3244%, .10초67.2385%, .20초72.0283%, .50초61.3583%였다. 나머지 premotor source의 기여도 남았다. 이는 선택 연결의 수학적 합이며 전체 BANC 전류·baseline 미래 반사실·CUDA 연산순서 exact 전류·생물학적 인과를 뜻하지 않는다. 모든 셀·전체101행·고정 세 구간 통계가 원자료에 있다.

실제 gpt-6-sol/high FIRST를 root31 tests/Ruff/별도6cell5edge 손계산으로 승인했고 같은Sol 수정은 필요하지 않았다. 저장 분석 감독2.5263281초/peak348864512bytes, root 독립 sparse matrix 재계산의 최대 차이2.273736754e−13(사전atol1e−9/rtol1e−12), 대응 원시 신호·12501개 통계·topology·해시·종료를 확인했다. Root 감사1.1126753초는 추가 원본0이다. Agent Ruff 사전 실패3건과 수정·전체 생성비용2.4165564초, root 생성감독1.6732505초를 보존했다.

다음 설계 검토는 기존24core로 selected outgoing 연결이 있는 source를 자원 소모에서 제외하는 구조다. 정적 연결 목록에서는 positive-only3194source 중1681개가 해당하고1513개/117973개 연결이 남는다. Source별 자원을 공유하므로 면제 source의 다른 outgoing도 함께 영향을 받는다. 이 정적 목록만으로 core rate나 미래 궤적 보존을 보장하지 않으며 아직 새 후보·원본 예약은 없다. u/tau의 작은 격자나 결과 기준 완화는 하지 않는다.

원본 누적8022.108435499544/8095초·원본 예약0·실행중0. 전체 F1–F8은 미완료다. M2/활성UI/F6미노출과 모든 과거 실패·비용을 유지한다.

증거: `verification/banc-natural-gait-m3-20260919/e295-core-resource-localization/`, `parent-e295-saved-audit.json`, `parent-e295-source-review.json`, `parent-e295-afferent-topology-inventory.json`.
