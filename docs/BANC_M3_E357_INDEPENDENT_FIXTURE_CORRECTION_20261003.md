# E357 ROOT 잔여 수정: 독립 입력 판정과 whole 재검증

전체 first actual27.476초/exit1/17passed·1failed/모든 소유 종료/cleanup0을 보존한다. 원본0/H356not_evaluated이고 current Q/F는 false다. 실패한 기존 ROOT linear positive fixture에서 T=Q 최대 오차2.108e-18, observed=quadratic exact 최대8.042e-18이다. 그러나 loss의 이차항 때문에 first-order reconciliation 오차6.0508825%가 고정5%를 넘었다. 따라서 분석기의 refuted는 맞고 기존 fixture의 conditional_support 기대가 틀렸다. 계산/원본/native/evaluator/5% 기준을 수정하지 않는다.

ROOT SECOND 이후 잔여 수정으로, 이전 independent source와 실패 증거를 그대로 두고 `parent-e356-root-e357-independent-cases-v2.py`를 만든다. 같은 두 함수 이름을 유지한다. 첫 함수 안에서 이전 near-cancellation 입력이 여전히 refuted 및5%초과인지 검사한 뒤, 동일한 unequal leg norms/실제float32 .001 방향/linear closed-form Jacobian에 고정 +.05 bias를 가진 별도의 deterministic positive 입력을 사전 정의해 conditional_support, T=Q, observed=Q_exact를 검증한다. 두 번째는 새 positive 입력의 출력 방향을 반전했을 때 refuted를 검사한다. 기존 반례를 지우거나 성공 기준을 완화하지 않으며 positive/negative branch를 모두 독립 계산한다. 이 생성 입력의 지지는 원본/자연 보행 지지가 아니다.

전체 재검증은 수정된 입력을 사용한 one new prospective v2, 같은16 mandatory+ROOT2/54 phases/native20/raw/gradient/정상 종료다. 동등한 실패 입력의 무수정 자동 재시도가 아니다. 첫whole27.476+v2 full outer최대36=63.476초로 계획 whole allocation65 안에 머문다. 새 generated-only witness/normalizer copies는 v2 exact C/command/prefix/original0/child33/full36을 강제하며 기존 original ADMIN와 이전 helper는 변경하지 않는다. 새 등록/ROOT reflection은 기존 source·metadata maps를 보존해 v2 source/helpers/actualreview만 prospectively 추가한다. Q65는 상한으로 유지하되 이번 actual36 계약을 함께 검사한다. 공유150초와 최소 마감5초, 원본9900/22+조건부3/재시도0 유지. 이후 원본 여부는 닫힌 current Q/F만 결정한다.

실제 Astra6/max가 이 계획과 numerical failure JSON을 source/JSON-only로 재검토한 후 ROOT가 반영·구현한다. E357 Sol FIRST 및 ONE R1은 이미 닫혔고 새 Sol 수정 라운드를 만들지 않는다. 실제 속도 개선/보행/생물학 완료 주장은 없다.
