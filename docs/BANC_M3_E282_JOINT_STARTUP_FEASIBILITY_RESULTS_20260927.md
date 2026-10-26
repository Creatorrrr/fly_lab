# E282 공동 시작 보정 결과 — 첫 선형 문제에서 기각

2026-09-27. [사전 설계 및 원본 전 보완](BANC_M3_E282_JOINT_STARTUP_FEASIBILITY_DESIGN_20260927.md)에 따라 여섯 다리의 중립 q/LTM와 두 Jacobian을 exact 재현했다. 다리별 q quota를 하나의 global q norm으로 바꿨지만 첫 공동 QP가 `PrimalInfeasible`로 끝났다. 보정된 실제 평가와 f32 평가를 시작하지 않았다. **유효한 진단·전체 초기화 기각**이며 자연 보행 목표는 미완료다.

Root는 저장된 원자료로 전체 P/A/b, 여덟 cone, 다리별 sparse Jacobian, f32 안쪽 계수 구간을 독립 복원했다. Dual cone와 계수 box로 계산한 가능한 dual slack 상한은 -0.999997385300이었다. f32 안쪽 제한을 제거하고 원래 f64 구간을 사용해도 -0.998763308330으로 남는다. 따라서 새 저장 정밀도 보완이 이 선형 실패의 유일한 원인은 아니다.

추가로 **RF의 q peak와 원래 coefficient/trust 제약만** 남기는 저장 dual 부분 인증을 구성했다. Global norm, 모든 LTM, 다른 다리 제약을 제거해도 중립 Jacobian의 가능한 peak 하한은 **.036805003121**이다. 이는 최종 내부 peak .0359964 및 명목 .036보다 크다. 1e-6 선형 여유값 제거로도 이 RF 부분 인증은 해소되지 않는다. 다리별 quota 또는 LTM 제약을 없애는 방법만으로 중립의 첫 선형 문제를 통과할 수 있다는 설명은 지지되지 않는다. 이 결과는 **중립의 고정 선형 근사와 ±.05 영역**에 관한 것이며 다른 비선형 시작점이나 BANC 모델 전체의 불가능성을 증명하지 않는다.

최대 dual은 RF local joint 0의 control 99와 반대 부호의 control 79–84에 집중된다. 개별 행만 따로 본 box 하한에서는 불가능한 행이0개였으므로 여러 시점의 요구를 동시에 맞추는 문제가 중요하다. 다음 단계의 의미 있는 질문은 RF를 포함해 더 완만한 중간 목표에 실제로 도달한 뒤 fresh Jacobian으로 고정 최종 목표에 접근할 수 있는가이다. 최종 기준이나 기존 실패를 바꾸지 않고 사전 설계가 필요하다.

Actual gpt-6-sol/xhigh FIRST75검사 → root FIRST → 같은Sol R1(원본 전 inward-f32 구간 보완) → root SECOND82검사/Ruff와 별도 analytic2solver를 완료했다. Root 잔여 source 수정은 없었다. Agent 생성 tiny15호출/162iter, root tiny14호출/156iter이며 원본과 분리 기록했다. R1 최초 test 수집 문법 오류2개도 비용에 포함했다.

원본은 engine6/max logical live1, 실제 평가6, forward/derivative 각각12,000ticks, QP1/13iter다. Constructor .119093700021초, solve1.768555300019초, certificate .047956600029초, engine creation .385836399917초다. Supervisor **4.211073899990879초**, RSS **664,231,936bytes**, 출력 **78,287,627bytes**. Owned PID33932/36656 종료를 확인했다. 누적 **7845.051662199552/7905초**, 잔여 **59.948337800447916초**다. Root raw 검산 .307237699977초, peak-only 저장 계산 .012495700037초이며 추가 solver/model/original0이다.

Teacher/objective/whole BANC/body/physics/fullfit/F6/후보 채택은0이다. E277 train/LM, E279 native 시작, E281 초기화 기각과 F1–F8/M2/E150/UI 상태를 보존한다.

증거: `verification/banc-natural-gait-m3-20260919/e282-joint-startup-feasibility/`, `parent-e282-original-final-review.json`, `parent-e282-peak-infeasibility-localization.json`, `parent-e282-output-manifest.json`, `parent-e282-final-source-review.json`, `parent-e282-original-supervisor.json`.
