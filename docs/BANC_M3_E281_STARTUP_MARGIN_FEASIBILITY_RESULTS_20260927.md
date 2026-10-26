# E281 시작 여유 초기화 결과 — 전체 초기화 기각

2026-09-27. [사전 설계](BANC_M3_E281_STARTUP_MARGIN_FEASIBILITY_DESIGN_20260927.md)의 동일한 여섯 중립 출력·Jacobian을 exact 재현했다. LF/LM/LH/RM은 두 번 보정과 float32 재평가 후 강화한 조건부 시작 기준을 통과했다. RF/RH가 실패하여 전체 초기화는 **기각**이다. 유효한 진단이며 학습·실제 보행·최종 F1–F8 통과가 아니다.

| 다리 | 최종 평가 | q peak | q SSE / 고정 quota | 판정 |
|---|---|---:|---:|---|
| LF | f32 | .035995435866 | .285093090781 / .298951328963 | 통과 |
| LM | f32 | .035993933308 | .067972855395 / .068046793251 | 통과 |
| LH | f32 | .035995377100 | .236973134408 / .236974099478 | 통과 |
| RF | 중립, 첫 보정 불가 | .039423500469 | .262285240681 / .218999918919 | 기각 |
| RM | f32 | .035995430528 | .085353447256 / .086351593954 | 통과 |
| RH | 두 번째 f64 | .038653626952 | .182565941642 / .181683551243 | 기각 |

q peak 한도는 .0359964다. 각 다리 LTM 한도는 통과했다. RF/RH는 f64 단계에서 실패했으므로 f32를 평가하지 않았다. 일부 다리의 성공을 전체 후보로 조립·채택하지 않았다.

## 실패의 범위

RF 첫 선형 문제는 `PrimalInfeasible`이다. Root는 저장된 dual과 정확한 계수 box를 이용하여 가능한 dual slack의 상한 -0.999980444525를 계산했다. Dual cone 조건도 만족하므로 **이 고정 선형 문제**의 불가능성을 지지한다. 모든 비선형 계수 해의 불가능성은 아니다. LTM norm dual을 제거해도 상한 -1.697795721734로 남았다. q norm dual을 제거한 기존 인증벡터는 양의 상한을 내므로 더는 불가능성을 증명하지 못한다. 이것이 q norm을 제거한 문제의 실행 가능성을 증명하는 것은 아니다. 가장 큰 peak dual은 RF local joint 0의 control 99에 있다.

RH 첫 보정은 선형 q peak .035995400000 대비 실제 .044471209288, 두 번째는 선형 .035995399964 대비 실제 .038653626952다. 선형–실제 차이의 max는 각각 .011947141638/.004260836963이며 두 번째 실제 위반 행은 3개다. 따라서 단순 float32 반올림 문제가 아니다. 첫 보정은 중립 ±.05 경계에 도달했다.

다리별 proportional quota는 전체 global RMS를 보장하기 위한 충분조건으로 도입한 방법상 제한이다. 네 통과 다리의 미사용 q quota가 존재한다. RF 실패를 전체 global RMS 계약의 불가능성으로 확대하지 않는다. 다음 설계에서는 이 인위적 다리별 할당과 비선형 보정 경로를 구분해야 한다. 실제 시작 기준 .04/.018, E277 train 20%·LM 개발 미달, E279 native 시작 기각은 그대로다.

## 검토·실행·자원

Actual gpt-6-sol/xhigh 구현 → root FIRST → 같은 Sol R1 수정 → root SECOND를 거쳤다. FIRST 74 생성 검사/Ruff, 별도 3개 analytic 문제, SECOND 변경된 runner 39 검사/Ruff를 통과했다. 고유 검사 범위는 86개이며 최종 버전 전체를 한 번에 86개 재실행한 것은 아니다. E280 실제 engine 생존 한도 보장 철회는 SHA로 명시적으로 계승했고, 새 runner의 실제 참조 해제는 weakref 회귀로 확인했다. E280 수치 결과를 실제 live-engine ≤2 준수로 재해석하지 않는다.

원본은 engine 6개/max logical live 1, QP 11회/215 iterations, 실제 조건부 평가 20회, forward 40,000/derivative 32,000 ticks다. 10개 보정 인증은 root가 원자료로 P/A/b, cone, stationarity, dual gap, 실제 제약 및 f32를 독립 검산했다. 원본 wall 6.3441338초, supervisor **7.0668290999601595초**, RSS 507,047,936 bytes, 출력 250,241,031 bytes다. Owned PID 21120/32208 종료를 확인했다. 누적 **7840.840588299561 / 7905초**, 잔여 **64.1594117004388초**다.

Root 원자료 검산은 passive 1.106411100016초, 추가 저장 dual/RH 검산 .028552100004초이며 원본·solver 추가 호출은 없다. 저장 NPZ에 없는 정수 metadata를 읽으려던 root localization harness의 `KeyError nonnegative_rows`와 수정 전 외부 wall .6015027초를 보존했다. 원본 결과에는 영향이 없다. 원본 인쇄 harness의 `KeyError status`, FIRST 중 오래된 draft `parameters`를 기대한 root schema harness 오류 및 Sol의 최초 analytic fixture 보정 비용도 이전 review/receipt에 남아 있다.

Teacher/objective/full fit/whole BANC/body/physics/F6/제품 후보 채택은 모두 0이다. 최종 목표는 active·미완료다.

원자료와 추적 자료는 `verification/banc-natural-gait-m3-20260919/`의 `e281-startup-margin-feasibility/`, `parent-e281-original-final-review.json`, `parent-e281-output-manifest.json`, `parent-e281-saved-rejection-localization.json`, `parent-e281-final-source-review.json`, `parent-e281-original-supervisor.json`에 있다. 이미 완료한 원본을 재실행하지 않는다.
