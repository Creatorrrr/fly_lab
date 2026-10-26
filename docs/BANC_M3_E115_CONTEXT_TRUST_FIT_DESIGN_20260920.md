# E115: 두 기록 입력에서 유지되는 작은 신경 가중치 보정

2026-09-20 · 부모 설계 · 구현: 기존 gpt-5.6-sol/xhigh

## 질문과 근거

E114는 네 문맥/가중치 조합의 CUDA 응답을 독립 계산으로 검증했다. E113 가중치는 E110 기록 입력에서 초기99 q최대 .03999597rad로 내부 여유 .04를 통과하지만, 자기 E113 입력에서는 .04613950rad로 여유가 없다. E110 가중치는 두 입력에서 원래 .05rad도 초과한다. 실제 E113의 F3/F4/F5 실패는 그대로이며 이 수치 검증은 보행 개선이 아니다.

이번 질문은 **현재 E113 가중치에서 작은 변경만 허용해도 두 기록 입력 각각의 초기 제약을 동시에 만족하면서, 두 입력 중 나쁜 쪽의 동작 형태 손실을 줄일 수 있는가**다. E112의 반복 횟수를 늘리는 재시도가 아니다. 검증된 두 문맥, 작은 변경 범위, 별도 제약과 최악 조건 목적함수를 사용한다. 실패해도 제약을 낮추거나 세 번째 입력을 만들어 다시 실행하지 않는다.

기록된 source/background는 새 후보의 실제 폐루프 입력을 예측하지 않는다. 통과한 조건부 후보가 있더라도 전체158706 신경과 신체의 별도 실제 검증이 필요하다. 외부 CPG/위상/궤적/root 보정은 최종 제어에 넣지 않는다.

## 고정 입력과 변경 범위

- E114 부모 fixture 및 검증 결과/원자료, E112b의 선택 calibration을 SHA로 고정한다. 각 문맥의 source와 **고유 W로 계산한 background**를 그대로 쓴다. 원래49 source/391 MN/1572 edge의 명시적 ID 매핑, 원래38tau 및 전체 생리,96근육 pool/moment/rest/LTM,10ms 근육 필터를 유지한다.
- 변경 변수는 E113의1572 efficacy에 대한 log 비율 delta뿐이다. 각 변수는 `[-log(1.1), +log(1.1)]`이고 원래 efficacy [.001,10]와의 교집합을 사용한다. 즉 원래 E113 대비 [1/1.1,1.1] 범위다. 부호, 새 edge, 신경 수, tau, 나머지 calibration 필드는 바뀌지 않는다. bounds 교집합이 비었으면 실행 전 실패한다.
- candidate W는 원래 signed source edge값 × 새 efficacy다. solver는 float64로 미분하되 최종 후보는 기존 native float32 저장 규칙으로 한 번 반올림하고, 실제 반올림된 W로 최종 검증한다. 최종 통과 판정은 float64 solver 예측만으로 대신하지 않는다.
- MN/근육/관절의 물리 다리별 분리를 기존 `leg_partition`으로 검증하고 여섯 블록을 독립 처리한다. 블록마다 같은 candidate W를 두 입력에 적용한다. `ContextResponse`는 두 pair에 독립0상태를 사용한다. 기존 helper/제품 코드는 수정하지 않는다.

## 목적함수와 제약

두 입력에 공통으로 bracket이 있는1823행(t=.5..9.61s)을 손실에 쓴다. 각 입력의 실제 E1 phase에 대응하는 기존 E85 주기 표적을 각각 쓴다. 표적 phase를 candidate 출력에 다시 맞추지 않는다. E114의 고정 E32 passive-context foot geometry를 사용하며 실제 발 경로로 부르지 않는다.

다리별·문맥별 손실 L_c는 기존 E112의 정규화 q, foot 위치, 인접5ms foot 속도, LTM(.2) 평균제곱오차 네 항의 합이다. joint/foot/velocity 정규화 척도는 원래 E112 fixture 그대로다. 각 문맥의 항과 합을 모두 기록한다.

solver 변수는 delta와 epigraph t다. 목적은 `t + 6e-5*(n_edge/1572)*mean(delta^2)`이며 제약 `t >= L_0`, `t >= L_1`을 각각 둔다. 두 문맥 손실을 평균내서 하나의 실패를 숨기지 않는다. t의 하한은0, 상한은없다. 초기 delta=0, t=max(초기 두 손실)+.01이고 이 초기 응답도 각 블록 평가 횟수에 포함한다.

각 문맥에서 원래 초기99행(.005..495s)에 대해 기존 내부 여유 네 값(qRMS .018, qmax .04, LTMRMS .009, LTMmax .025)을 각각 적용한다. 총8개의 제약을 평균/합으로 대체하지 않는다. 제곱 비율을 사용하고 solver cap=(1-2e-4)^2, selector cap=(1-1e-4)^2는 E112와 같다. 원래 .02/.05/.01/.03 판정도 별도로 보고한다. 각 다리 제약을 독립 만족해야 하므로 전체평균에 의한 희석이 없다.

SLSQP는 각 블록에 정확히 한 번, maxiter500/ftol1e-9로 실행한다. objective와8 startup+2 loss 제약의 정확한 autograd 미분을 공유한다. 캐시는 delta의 비트열로 조건부 응답/미분을 공유하고 t의 단순 항은 재계산한다. t만 바뀔 때 신경 응답을 다시 계산하지 않는다. epigraph t가 잘못된 과거 record를 후보 선택에 쓰지 않고, raw delta에서 실제 max(L_0,L_1)+prior와8개의 startup ratios로 선택한다.

각 블록은 유한한 best-any와 모든8 selector 제약을 통과한 best-feasible을 따로 보존한다. 후자의 실제 worst-context 목적이 최소인 record를 선택한다. 어떤 다리라도 feasible record가 없으면 전체 후보를 합성하지 않고 `no_joint_feasible_candidate`로 종료한다. best-any나 초기값을 조용히 섞지 않는다.

## 실행 전 검토와 한 번의 실행

부모 설계→Sol 구현→부모 첫 리뷰→필요한 첫 수정은 같은 Sol→부모 두 번째 리뷰→남은 수정은 부모 순서다. CUDA 미분, 제약 최적화, 신경 매핑 및 실패 회계가 결합된 작업이므로 실제 기존 Sol의 xhigh를 유지한다. followup으로 effort를 바꿨다고 기록하지 않는다.

구현 범위는 새 `tools/fit_c_banc_m3_context_trust.py`, `tests/test_c_banc_context_trust.py`다. 모델 없는 작은 CPU 검사로 두 문맥 제약, epigraph 미분, delta/t 캐시 구분, 실제 float32 후보, empty bounds, selected ID 순서, raw/ledger/원래 예외 보존을 검증한다. 검증된 E114 helper와 E111 geometry를 읽기 전용 재사용한다.

부모 최종 소스 검토 후에만 RTX4080/cuda:0/Torch float64, OMP/MKL/OPENBLAS/Torch CPU thread1로 실행한다. 각 다리28벽초/500개 unique delta, 합3000개, 전체 fit180초, 내부프로세스210초, 외부240초다. 시간 제한은 각 평가와 Jacobian/기록 경계를 포함하며 더 작은 한도를 우선한다. 중단도 기록하고 자동 재시도하지 않는다. 블록 하나의 delta 평가는 두 개의2001행(10조건부초) **단일 다리** 응답이다. 시작/완료/저장·캐시 hit를 다리별로 기록하고 이를 전체 native 실행과 혼동하지 않는다.

초기 전체 응답은 E114 pair1/3 저장값을 읽는다. 각 블록 최초 delta=0의 출력을 해당 저장값의 다리 부분과 current<=1e-8, motor/activation/q/tendon<=1e-9로 검증한다. 서로 다른 float64 signed*efficacy와 native float32 초기 W의 차이가 생기지 않도록 solver의 초기 W는 **E113 실제 float32 W**에서 출발하고 selected edge에 `W113*exp(delta)`를 적용한다. 최종 efficacy와 native W의 저장은 위의 원래 source×efficacy 규칙이며 그 반올림 차이도 raw로 보고한다.

모든 다리에서 feasible을 찾았을 때만 합성된 실제 float32 W로 전체391MN의 두 문맥 응답을 CUDA batch 한 번 계산한다(2 responses/20조건부초). 부모 독립 검산도 새 최종후보의 두 문맥을 NumPy로 각각 한 번만 허용한다(합20조건부초). 기존 baseline을 재실행하지 않는다. solver 블록 응답과 마지막 전체 응답은 별도 계상한다. 실제neural/physical0이며 이 단계는fit1회다.

기존 E113 프로세스 체인의 보수 합589.5445802초와 E114 실제프로세스8.125초는 보존한다. 그 합597.6695802초를598초로 올려 이번 프로세스240초를 더한 별도 누적 실행 상한838초를 기록한다. 과거 E112506/E113630 상한과 실패 판정은 소급 변경하지 않는다. 부모 데이터 준비/코드리뷰/NumPy 검산 비용은 기존 개별 ledger와 함께 별도로 계속 기록한다.

## 최종 판정과 후속 분기

각 입력에서 초기99 원래/내부여유, 원래 E110 고정1828행 표적, 각 입력 own-phase 전체 유효 구간, 공통1823행의 q/foot/velocity/LFrow6/per-leg 값을 모두 저장한다. 공통 표적과 다른 mask의 loss를 같은 숫자로 섞지 않는다.

최종 float32 후보가 **두 문맥 모두** 내부여유를 통과하고, E114의 같은 입력/E113 W 대비 공통 구간 foot RMS와 velocity RMS가 각각20% 이상 감소하며 qRMS는5% 이상 악화하지 않을 때만 `robust_local_supported=true`다. 두 문맥 중 어느 한 조건의 미달은 실패다. E112의 다른 relative/absolute shape 결과도 적용 가능한 원래 정의 그대로 별도로 보고하며 기존 실패를 덮어쓰지 않는다. 목표 F3–F5나 자연스러움은 이 단계에서 판정하지 않는다.

조건부 screen을 통과하면 다음 실제10s+.1s 저장 재현 검증을 소스검토·자원·기준을 별도로 동결한 뒤 단 한 번 수행한다. 실패하면 같은 보정의 횟수/범위를 늘리지 않고, 초기와 지속 구간의 모집 제약 또는 신경 정지 경로 간섭 등 별도 가설을 원자료로 판단한다. 신선한 F6 초기상태는 노출하지 않는다. M2의263개 및 E110/E113/E114 원래 산출물은 불변이며 새 출력 `e115-context-trust-fit/`만 사용한다. 전체 F1–F8 목표는 active다.
