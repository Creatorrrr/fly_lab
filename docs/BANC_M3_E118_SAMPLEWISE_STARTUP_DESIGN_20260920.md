# E118: 모든 초기 sample의 CUDA 미분과 저장 방향 검증

2026-09-20 · 부모 설계 · Sol 구현/첫 수정 · 부모 재리뷰 뒤 잔여 직접 수정

## 목적과 범위

직전 목표 턴은 progress다. E117의 단일6pair CUDA 응답과 독립6pair NumPy 검산으로 E116 두 방향의 비선형 실패를 확인했다. 현재 peak 하나에 대한 제곱오차 affine 제약은 다른 초기 sample의 증가를 막지 못했다. 전체 F1–F8 목표 및 실제 E113 F3/F4/F5 실패를 유지한다.

이번 단계는 E117 selected_solver의 **저장 응답만** 사용해 초기99 전체 관절/LTM signed residual의 delta 미분을 계산하고 검증한다. E115에 저장된8개 startup gradient 및 부모 독립 순차 필터와 비교한다. E116의 두 고정 h를 전체 signed sample 선형 예측에 적용해 E117 실제 비선형 결과와 비교한다. 새 모델 응답·fit·LP·native/body·calibration·F6 실행은0이다. 여기서 새 candidate를 생성하거나 채택하지 않는다.

사전 구조 검사 parent-e118-structure-review.json의19개 검사를 통과했다. 각 다리의 선택 edge post가 다른 다리 관절이나 LTM에 주는 직접 계수는 정확히0이며, 원래 edge mask와0..99 시간축을 유지한다. 이 검사는 배열 대수이며 미분 필터나 신경 응답이 아니다(내부 .1217468초/외부 .5141221초).

## 고정 입력과 미분 정의

E117의 parent-final-review/원본manifest/response-raw/metrics-raw, E114 fixture, E115 선택 delta 및 여섯 선택 eval, E116 두 저장 h를 부모 freeze로 고정한다. E117의 input/source freeze와 read-only reviewed_inputs를 재사용할 수 있다. 원래391 MN/49 source/96 muscle/42 joints/6 LTM/1572 selected edges, 두 문맥 E110/E113, 정확한 ID 순서와 signed W를 유지한다. E117 response의 selected_solver는 앞2pair이며 W는 weights[0]이다. 기존 source/background와 실제 기록된 current/activation을 사용하고 다시 모델을 실행하거나 background를 trial W로 분리하지 않는다.

0..99행(100행, 5ms, 끝 .495초)에서 edge e=(r,c)의 전류 미분은 `dC_e=S_c*W_e`다. 저장 current에서 `u=a*(C-threshold)/cap`, `f=relu(cap*tanh(u))`이며 `df_e=1[cap*tanh(u)>0]*a*(1-tanh(u)^2)*dC_e`다. ReLU0의 미분은 PyTorch와 같은0이다. a/cap/tau는 해당 post MN의 값이며 W_e는 selected_solver float64 값이다. delta를 바꾸거나 float32로 반올림하지 않는다.

모든 edge/문맥을 독립 channel로 두어 df를 기존 linear_response(dt=.005,tau_post), endpoint_response(dt=.005,tau=.01)에 차례로 적용한다. helper별 호출은1회씩이며2문맥×1572edge=3144독립 channel의 한 derivative bank다. 원래 state의 미분은0이다. 초기 prefix의 인과성을 사용하며, 전체2001행에서 계산된 E115 gradient와 비교해 FFT roundoff를 검사한다. ContextResponse.evaluate, LegStage.tensors 또는 다른 forward response는 실제 runner에서 호출하지 않는다.

edge별 muscle filter 결과를 z_e라 하면 `Jq[j,e]=.02*(moment@pool)[j,r]*z_e`다. `JLTM[e]=.02*(1-tanh(activation_LTM/10)^2)*pool[LTM,r]*z_e`다. 같은 .01초 muscle filter가 선형이므로 pool과의 계산 순서를 바꿀 수 있다. 출력은1..99행이며 다리별 원래 edge mask만 저장한다: Jq(2,99,7,n_edge_leg), JLTM(2,99,n_edge_leg). 모든 다른 다리 계수0 및 ID/축을 재검사한다.

## 원자료와 비교

derivative-bank-raw.npz에 current_derivative/forcing_derivative/motor_derivative/activation_derivative(2,100,1572) 및 edge IDs/rows/cols/selected weights/문맥/시간축을 먼저 저장한다. 여섯 jacobian-{leg}.npz에는 위 Jq/JLTM, 대응 edge positions/IDs, 원래 signed errors(2,99,7)/(2,99), reconstruct된8×n startup-gradient와 E115 reference를 저장한다. 모든 이용 가능한 tensor/metadata는 assertion이나 실패 가능한 후속 receipt 전에 복구 가능해야 한다.

qRMS² gradient는 `2*sum(error*J)/(99*7*inner_qRMS^2)`, LTM RMS²는99개 평균이다. qmax²와 LTMmax²는 각 문맥의 실제 최대 제곱오차 sample을 사용한다. 정확히 동률인 sample은 PyTorch amax 미분과 같이 평균한다. 정확 동률과 기존1e-10 상대 설명용 동률 수를 따로 기록하며 gradient의 동률을 임의의 tolerance로 합치지 않는다. 재구성8행의 순서는 E110[qRMS,qmax,LTMRMS,LTMmax], E113[동일4]다.

E115 저장 jacobian[2:]와 전 edge 최대 절대 차이1e-8 이내여야 한다. reconstructed startup ratios와 E117 baseline도1e-8 이내로 검사한다. 수치검사 전에 reference/difference/raw를 보존한다. 이 비교는100행 derivative와 기존2001행 autograd의 일관성을 검증하며 새로운 full response가 아니다.

고정 h=E117 steps.increments의 shape_box/startup_box 각각을 J에 곱한다. selected_solver(h=0)까지3variant의 `error+J*h`와 여기서 다시 계산한 qRMS²/qmax²/LTMRMS²/LTMmax²를 기록한다. 현재 peak 제곱값의 affine 예측(E117 metrics의 affine_startup_ratios), 전체 signed sample 예측, E117 실제 nonlinear ratios를 나란히 저장한다. 새 최대 sample 위치/시간/global joint와 baseline 위치 변경을 기록한다. 선형 signed error를 제곱해 생기는 값은 음수일 수 없으며, 기존 음수 affine 예측을 덮어쓰지 않는다. 원래99/내부 제약 판정은 설명용으로만 보고하고 실제 candidate 승인에 쓰지 않는다.

## 구현·리뷰·실행 한도

새 tools/analyze_c_banc_m3_samplewise_startup.py와 tests/test_c_banc_samplewise_startup.py만 Sol이 작성한다. pure tensor helper와 runner를 분리된 함수로 구성한다. 재사용 source, calibration, UI, 기존 결과는 수정하지 않는다. 신경 미분·CUDA·다중 축·제약이 결합되어 기존 명시적 gpt-5.6-sol/xhigh를 유지한다. 부모 첫 리뷰 뒤 첫 수정은 같은 Sol, 두 번째 리뷰 뒤 잔여 수정은 부모가 수행한다.

합성 CPU 테스트는 작은 실제 ContextResponse autograd와 analytic derivative를 비교한다. 두 문맥, 비정렬 edge, 억제/흥분 signed weight, 양/음 forcing 및 ReLU0, 서로 다른 tau, 정확 peak 동률, prefix/미래 입력 독립성을 검증한다. 실패 주입으로 입력 준비 전 계상0, 필수 시작 receipt 실패 시 실제 filter0, 반환 이후 실패 시 모든 알려진 metadata/수치/원래 예외 보존을 검사한다. 실제 데이터 계산은 최종 소스 리뷰 전0이다.

이후 RTX4080 cuda:0 float64, OMP/MKL/OPENBLAS/Torch CPU thread1에서 단일 원래 derivative bank만 실행한다. 모든 입력 tensor 준비 후 시작 receipt를 먼저 성공시킨다. bank started/materialized/raw_saved와 여섯 jacobian materialized/raw_saved를 따로 계상한다. derivative context2/channel3144/prefix100/.495초를 기록하되 조건부 응답/시뮬레이션 시간으로 부르지 않는다. 선형 필터와 endpoint 필터 각1회, model response/native/body/LP/fit/calibration0이다. 출력 e118-samplewise-startup/는 존재하면 거부하며 재시도0이다.

내부15/외부25초. E117의709.5144862초를710으로 올린 별도 보수누적735초 한도를 사용한다. 과거 E115/E116/E117 한도나 실패를 소급 변경하지 않는다. 부모 CPU 준비/검토는 별도로 기록한다.

원자료 생성 후 부모는 독립 NumPy 순차 미분 필터를 한 번 계산한다(동일2문맥/3144channel/100행, 내부15/외부20초). 본래 model response와 실제 신경/신체는0이다. 각 bank/다리 Jacobian의 최대 절대 차이1e-9, 재구성 startup gradient의 차이1e-8을 검사한다. 부모는 저장 signed 예측/peak/ratio/판정 및 자원/출처를 독립 감사한다. M2263/E110201/E113207/E11419/E1153043/E11665/E11720 원본과 F6 미노출을 보존한다.

검증에 실패하면 원본을 남기고 새 실행/한도 확대 없이 원인을 분석한다. 통과하면 전체 sample 제약과 RMS 제약, trust step, 실제 nonlinear acceptance를 함께 쓰는 다음 최적화를 설계한다. 전체 목표 완료 또는 자연스러운 생물학적 보행의 증거로 이 미분 검증을 대체하지 않는다.
