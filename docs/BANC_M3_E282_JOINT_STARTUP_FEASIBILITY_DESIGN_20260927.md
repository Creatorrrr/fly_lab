# E282 — 여섯 다리 공동 시작 여유 보정

2026-09-27 root 사전 설계. M3 finite-goal, F1–F8, E277 train/LM 실패와 E279 실제 시작 실패 및 E281 전체 초기화 기각을 유지한다. 최종 목표는 자연스러운 BANC 단독 보행이며, 본 진단은 다음 학습을 위한 시작점의 실행 가능성을 판별한다. 결과와 무관하게 full fit/native/body/F6/제품 후보 승격은 이 단계에 포함하지 않는다.

## 선택 이유와 구별할 질문

E281 RF의 고정 첫 선형 문제는 저장 dual/box 검산으로 불가능성이 지지된다. LTM norm dual을 제거해도 불가능성이 남지만 q norm dual을 제거한 기존 인증은 불충분하다. E281의 다리별 proportional q quota는 전체 RMS 보장의 충분조건이었고, LF/LM/LH/RM 성공점에는 미사용 quota가 있다. 이 수치들은 local quota가 장애일 가능성을 지지하지만 global 문제의 실행 가능성을 증명하지 않는다. RH는 선형–실제 차이가 남아 두 보정 후 실패했다.

고정 quota를 유지하는 중간 목표 continuation도 가능한 대안이나, RF의 인위적 norm 분배 제한을 유지한다. 따라서 먼저 **여섯 다리 공동 q norm과 fresh-J 보정**을 선택한다. 가설 H282는 이 공동 문제에서 중립 ±.05 안의 최종 시작 여유를 찾을 수 있다는 것이다. 경쟁 설명은 그 영역의 공동 제약 자체가 불가능하거나 비선형 응답 때문에 제한된 보정으로 도달하지 못한다는 것이다. 이것은 고정 quota 제거와 최대 네 보정을 결합한 방법의 진단이며, 성공해도 각각의 기여를 분리해 주장하지 않는다.

## 불변 조건과 바뀌는 계산

E281이 SHA로 고정한 모든 원본 context/target/engine/decoder/physiology/51 source/1,660 기존 edge를 그대로 사용한다. 원본 teacher 자료 평가나 새 데이터 노출은 없다. E280 수치근거는 실제 engine 생존 보장 철회 qualification과 함께 계승한다. frozen E277–E281 source는 수정하지 않는다.

LF/LM/LH/RF/RM/RH의 coefficient triple을 leg-major로 연결한다. roster의 실제 edge index와 slice를 저장하며 1,660개 edge 중복·누락 및 q 42/LTM 6 축 중복·누락을 검사한다. 이것은 최적화 내부 배열이며 아직 제품용 global edge 순서의 후보 파일을 생성하지 않는다. 각 다리의 qerr (99,7), lerr (99,1), J를 같은 순서로 연결한다. 각 조건부 엔진의 독립성은 기존 source leg ownership 검증을 계승한다. joint Jacobian은 sparse block diagonal을 사용하고 큰 dense global Jacobian을 만들지 않는다.

최종 실제 조건부 검사는 다음을 동시에 요구한다.

- 모든 coefficient [0,8], triple sum [.125*(1+1e-6), 8*(1-1e-6)], **원래 중립 기준** 각 coefficient ±.05.
- 모든 q peak ≤ `.036*(1-1e-4) = .0359964`.
- 전체 q SSE ≤ `4158*(.0162*(1-1e-4))**2 = 1.091007285808255`. 다리별 q SSE는 보고하지만 proportional quota를 요구하지 않는다.
- 모든 LTM peak ≤ `.025*(1-1e-4)` 및 각 다리 LTM SSE ≤ `99*(.009*(1-1e-4))**2`. E281 LTM 조건은 그대로다.

이것은 E281과 동일한 global q RMS와 peak를 보장하는 방법 변경이다. 원래 실제 native .04/.018 기준을 낮추거나 수정하지 않는다. 내부 목표에 작은 tolerance를 더해 수용하지 않는다.

## 고정 알고리즘과 판정

1. 중립에서 각 다리를 한 번 실제 startup101 엔진으로 평가하고 E281 저장 중립 q/LTM abs1e-10, 양쪽 J abs1e-9를 재현한다. 각 evaluation마다 독립 새 엔진을 생성·종료하되 실제 참조를 해제한 뒤 다음 엔진을 생성한다. 이 재생성은 context state를 바꾸지 않으며 `solve`가 저장 조건부 초기상태에서 시작하는 기존 의미를 유지한다.
2. 현재 여섯 실제 결과를 묶어 최소 norm `.5*z.T*z` 문제를 구성한다. `P=I, q=0`, n=4,980. coefficient/triple/중립 trust/개별 보정 max .05와 모든 peak, 하나의 global q SOC, 여섯 개의 per-leg LTM SOC를 쓴다. Peak와 norm radius에서만 선형 여유 1e-6을 더 뺀다. 원래 중립 trust box를 보정마다 재설정하지 않는다.
3. Clarabel 0.11.1, max100iter/thread1/tol1e-8, 호출당 최대4초 및 남은 science deadline 중 작은 값을 쓴다. 정확한 원문제 `b-A*z`, dual cone ≥0, primal/residual ≥-1e-8, stationarity infinity ≤1e-8, primal .5*z², dual `-.5*||A.T*y||²-b.T*y`, 상대 gap [-1e-10,1e-8], complementarity1e-8를 별도 인증한다. Solved/AlmostSolved만 후보 보정으로 허용한다. 상태·원문제·원시 x/s/y와 certificate를 저장한다. 원시 결과를 clip·repair하지 않는다.
4. 인증된 보정이 실제 coefficient 조건도 통과하면 각 다리 startup을 순차 평가해 실제 응답과 fresh J를 다시 얻는다. 위 실제 최종 조건이 모두 통과하면 보정을 끝낸다. 미달이면 새 실제 Jacobian에서 다음 공동 보정을 구성하되 **총 네 번**까지만 한다. QP 인증 실패/명확한 infeasible에서 중단하며 별도 line search·quota 재할당 sweep·warm restart·다섯 번째 보정은 없다. 자료·계산·저장·source·deadline 실패는 invalid/incomplete와 부분 raw/cost를 보존한다.
5. f64 전체 통과 후에만 동일 공동 coefficients를 한 번 f32→f64로 반올림한다. clip 없이 coefficient/trust를 먼저 검사하고 여섯 startup을 derivative=False로 순차 재평가한다. 전체 q/per-leg LTM 실제 조건을 다시 충족할 때만 `initialization_supported`다. f64/f32 둘 중 하나라도 미달하면 기각이다.

공동 첫 선형 문제의 valid solve 여부와 실제 peak/SSE 궤적이 진단 결과다. 최종 6/6 f32 공동 기준 성공만 다음 full-fit 설계에 사용한다. 실패는 이 공동 방법·고정 중립 영역·네 보정 한도 내의 실패이며 모든 비선형 source 모델 불가능성으로 확대하지 않는다. Solver 시간 한도는 수학적 infeasibility로 해석하지 않는다. 이전 E281 성공 네 다리만 보존·고정하거나 실패 두 다리만 바꾸는 최적화는 하지 않는다.

## 구현·검토와 자원

Actual gpt-6-sol/xhigh가 신규 helper/runner와 두 대응 test만 구현한다. Root FIRST → 같은 Sol 첫 수정 → root SECOND → root 잔여 수정 순서를 따른다. Helper의 sparse 문제, certificate, 실행 흐름과 엔진 수명/오류 비용/strict f32 검사를 생성 CPU 문제로 확인한다. Agent 생성 테스트는 CPU wall180초/real tiny solver60회 이하, n≤6으로 제한한다. Root 검사는 별도 기록한다. 원본 문제·CUDA 호출은 root source 승인 및 freeze 전에는 0이다.

원본은 한 번, science40초/cleanup15초/supervisor55초, RAM6GiB/GPU 추가4GiB/출력2GiB/free disk4GiB다. 공동 QP≤4회, engine≤36개, 실제 live engine≤1, startup evaluation≤36 (중립6+보정24+f326), forward≤72,000ticks/derivative≤60,000ticks다. 모든 요청·부분 완료·iterations·constructor/solve/cert/engine 비용과 cleanup 실패를 보존한다. Worker CUDA 작업은 직렬이며 별도 GPU sweep은 없다. Root 외부 supervisor가 실제 process tree RSS/시간과 owned PID 종료를 확인한다.

현재 원본 누적 7840.840588299561/7905초, 잔여64.1594117004388초 안에서55초를 배정한다. 배정 증액·비용 초기화는 없다. 생성 CPU·수동 saved-array 검산 비용도 별도로 기록하며 무료 원본으로 취급하지 않는다. Source 구현 중 원본 예약은0이고, root 최종 source 승인·현재 GPU/디스크/PID 확인 뒤 정확한55초 예약과 source/input SHA를 고정한다. 일치하지 않으면 원본을 시작하지 않는다.

기존 M2/E150, 실제 켜진 UI, F6 미노출과 모든 실패 원자료를 보존한다. 실험 결과는 active plan/progress/checkpoint에 기록하고, 다음 결정은 이 결과로 고른다. 구현 테스트 또는 조건부 초기화 성공만으로 goal complete를 선언하지 않는다.

## 원본 전 root FIRST 보완 — f32 표현 가능한 계수 구간

생성75검사/Ruff와 별도 root analytic tiny2문제는 구현의 기존 수학을 확인했다. 다만 원래 neutral trust 경계의 .95는 f32 변환 후 .949999988079071, .05는 .05000000074505806이 되어 엄격한 원래 구간을 벗어난다. E281 RH도 중립 변위 .049999995에 도달했다. 이는 원본 E282를 실행하기 전에 확인한 표현상의 문제이며, E282 실제 성공/실패를 보고 판정 기준을 바꾸는 것이 아니다.

Root FIRST R1로 다음 수치적 보완을 같은 Sol에 요청한다. 각 QP의 **새 coefficient 값**에 대해 원래 [0,8]·neutral±.05·current±.05 교집합의 하한/상한을 먼저 계산한다. 각각 그 구간 안의 가장 가까운 f32 표현 값(하한은 ceil-f32, 상한은 floor-f32)을 float64로 저장하고 이 두 값에서 current를 빼 선형 step bounds로 쓴다. 반올림이 이미 안쪽이면 추가 ULP 이동을 하지 않는다. directed nextafter는 구간 계산에만 사용하며 원시 solver 결과나 선택된 coefficient를 clip/repair하지 않는다.

인증 후 실제 평가 전에 계산된 새 coefficient가 이 안쪽 구간도 **허용오차 없이** 만족해야 한다. 인증 tolerance 안에서 살짝 벗어났더라도 보정·클리핑 없이 기각한다. 이 검사는 최종 f32 변환이 original coefficient/trust 구간을 벗어나지 않도록 하는 것이며, f32 실제 q/LTM 재평가는 계속 필수다. 원래 f64/f32 실제 제약과 global peak/SSE, 중립 trust, 네 보정 상한과 모든 자원 한도는 동일하다. 원본 예약/실행은 여전히0이다.
