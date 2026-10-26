# E114: 후보별 기록 입력을 보존하는 조건부 검증

2026-09-20 · 부모 설계 · 구현 담당: 기존 명시적 gpt-5.6-sol/xhigh

## 판별 질문과 범위

E113은 실제 F3/F4/F5 미달이며 채택하지 않는다. 고정 E110 입력 예측은 실제 E113과 전체 qRMS .0277491rad 차이가 났지만, 기록한 E113 실제 전류를 같은 조건부 식에 넣으면 .00042644rad다. 부모의 정적 감사에서 MN391→source49의 직접 유효 연결156개, 두 단계 내49개 전체 도달을 확인했다. 1/1.1/10초 external drive 차이는 감각624개에만 있었고,1.1초의 큰 MN background 차이는 기존 억제성 IN19A003(83877)의 활동 차이가 주로 만들었다. 이 대수 분해는 감각과 재귀의 인과 기여를 분리하지 않는다.

다음 보정 전에 **기록 입력 문맥과 평가할 가중치를 명시적으로 분리하는 재사용 가능한 CUDA 조건부 응답**을 구현한다. E110/E113 각각의 실제 S,C,W에서 B=C−SWᵀ를 얻고 자기 응답과 교차 응답을 모두 평가한다. 이 단계에서는 효능 보정, 새 후보, 전체 신경/신체, 신선한 F6 상태를 실행하지 않는다. E112 실패나 E113 실제 실패와 그 기준/비용은 바뀌지 않는다.

## 불변 데이터와 계산

- 부모 `parent-e114-context-inputs.npz/.json`은 원래49 source/391 MN/1572 edge의 명시적 ID 순서, 원래38tau를 포함한 MN 생리,96근육 pool/moment/rest/LTM를 보존한다. 각 입력의 실제 float32 W를 float64로 올린다. `cal.edge_ids`는 정렬됐다고 가정하지 않고 ID 사전으로 대응한다.
- 두 입력은 t=0..10s,2001행,dt=.005s다. source와 전체 recurrent current 및 activation의0행은 저장된 원래 zero start다. MN의 직접 drive/adaptation은 원본 기록에서0임을 확인한다. 실제 command/tendon은 원본 trace/activation과 대응한다.
- 각 문맥의 고유 background는 반드시 그 문맥의 W로 뺀다. 다른 후보 W로 다시 빼서 후보 효과를 제거하면 안 된다. 파일 SHA를 고정하고 원본 metadata55와 source49 ID 대응을 검증한다. float64 재계산의 합산 순서 차이는 E112b에서 검토한 gamma 경계를 쓰거나 동일한 수학적 상한으로 raw를 먼저 기록한다. stored background를 수정하지 않는다.
- 수식은 기존 conditional response 그대로다: `I = S_context W_candidateᵀ + B_context`, `forcing=relu(cap*tanh(a*(I-threshold)/cap))`, zero-state piecewise-linear5ms forcing 응답(원래 tau),10ms endpoint muscle filter, `q=rest+.02 activation momentᵀ`, `LTM=.2 tanh(activation_LTM/10)`.
- 새로운 구현은 여러 문맥/후보 pair를 독립 batch 축으로 처리한다. 시간축을 연결하지 않고 각 pair가 독립적인0상태에서 시작한다. native network와 다른 endpoint 근사임을 명시한다. Torch float64 CPU/CUDA 및 candidate W에 대한 autograd를 유지하고, 공용 기존 helper/제품 코드는 수정하지 않는다.

## 고정 검증 실행

부모 소스 검토를 통과한 뒤 한 번의 명령으로 pair 순서 `(E110 input,E110 W)`, `(E110 input,E113 W)`, `(E113 input,E110 W)`, `(E113 input,E113 W)`를 하나의 CUDA batch로 계산한다. RTX4080/cuda:0을 확인하고 OMP/MKL/OPENBLAS/Torch CPU thread1을 쓴다. 새 batch 계산은4개의 조건부 응답(각10s,합40 conditional model seconds)으로 계상한다. 기존3개 기준 응답은 파일에서 읽으며 다시 실행하지 않는다:

1. E110/E110: `parent-e112-latest-conditional-baseline.npz`
2. E110/E113: `e112-dense-foot-recruitment-v2/prediction.npz`
3. E113/E113: `parent-e113-actual-input-response-raw.npz`

기준3개의 current 차이≤1e−8, motor/activation/q/tendon 차이≤1e−9를 검증한다. 원래 기록 대비 self-response도 전체/원래99별로 보고한다. 신체 미끄럼이나 F3–F5는 새로 계산/판정하지 않는다. 별도 부모 NumPy 검산은 아직 없는 E113-input/E110-W 응답1회만 허용한다(10 conditional seconds/0 actual). 합5개50 conditional seconds와 CPU 합성 단위검사의 tiny 응답은 분리 기록한다.

각 pair에 대해 다음을 원자료와 함께 보고한다:

- 원래99(.005–.495s)의 q/LTM RMS와 최대 오차. 원래 기준 .02/.05rad, .01/.03 유지. 여유 기준 .018/.04rad, .009/.025도 별도로 유지한다. 실패를 평균으로 숨기지 않는다.
- 원래 E110의1828개 고정 시각/표적에 대한 q, command 기반 fixed-context foot, 인접5ms foot velocity, LF row6, per-leg q 오차. 고정 시간축 비교다.
- 각 문맥의 실제 E1 peak를 같은 E85 주기 궤적에 맞춘 own-phase 표적 비교. E110 원래 표적과 일치를 먼저 확인한다. E113 bracketed mask/count/endpoints와 두 문맥 공통 mask를 모두 기록한다. 서로 다른 표적/마스크를 같은 loss로 혼동하지 않는다.
- E113과 E110 own-E1 위상 변화의 signed unwrapped 차이 및 원형 차이, 각 다리 RMS/최대. 위상 재정렬이 시간축 오차를 숨기지 않게 한다.

E111의 고정 E32 passive-context geometry만 사용한다. 이 foot은 측정한 실제 발 경로가 아니라 command에서 계산한 기준 기하학이다. 기존 E112/E113의 실제 기록·계산·원래 F 판정은 그대로다.

## 유효성, 비용, 실패 보존

순수 helper의 작은 CPU 검사는 batch 독립0상태, 다른 문맥 W를 background에 사용하는 오류, ID 순서/중복/NaN/불연속 시간, 실제 signed float32 W, autograd, raw/실행 오류 보존을 확인한다. helper의 응답과 같은 구현을 복제한 테스트만으로 수치 일치를 주장하지 않는다. 원래3개 독립 응답과 부모 네 번째 NumPy 응답이 실제 자료의 기준이다.

실행은 외부60벽초, 내부50벽초, 준비된 데이터에서 한 번의 CUDA batch1개(4 responses), fit0/실제neural0/physics0이다. 부모 데이터 준비 및 리뷰/NumPy 응답의 비용은 별도로 기록한다. 이 새 자료의 검증 단계는 이미 종료된 E113450+180≤630 예산을 연장하지 않는다. 전체 과거 비용을0으로 초기화하지 않는다. 자동 재시도/부족한 예산 연장 금지. 오류/미완료 시 raw와 started/completed/saved counters 및 실제 벽초를 남기고 원래 예외를 보존한다. raw를 저장한 뒤 수치 assertion을 한다.

모든 input/design SHA와 실제 runner/source/test SHA를 기록한다. 원래 M2의263파일과 E110/E113 원본 outputs는 불변이다. 출력은 새 `e114-context-validation/`만 사용하며 기존 파일을 덮어쓰지 않는다.

## 개발·리뷰·다음 판단

현재 부모 세션이 fixture/설계를 작성하고, Sol은 새 `tools/banc_m3_context_response.py`, `tools/verify_c_banc_m3_e114_contexts.py`, `tests/test_c_banc_context_response.py`만 구현한다. 기존 helper를 읽기 전용으로 재사용한다. 부모 첫 리뷰의 필요한 첫 수정은 같은 Sol, 두 번째 리뷰 뒤 남은 수정은 부모가 수행한다. 이번 일은 후보별 행렬 대응·CUDA batch 미분·동역학 근사 검증이 결합돼 기존 Sol의 xhigh가 적절하다. followup이 effort를 바꿨다고 기록하지 않는다.

E114는 보정 도구의 검증이고 보행 개선 성공이 아니다. 결과를 본 뒤 robust fit이 타당하면 두 문맥 각각의 original99와 형태 손실, 실제 W 변경폭, 다음 단 한 번의 actual body 검증을 실행 전에 새로 고정한다. 초기 또는 형태 제약의 구조적 충돌이 남으면 근육 모집/관절 전달 등 다른 가설을 그 근거로 설계한다. F1–F8 전체 목표는 active다.
