# E118 다음 설계안: 초기 모든 sample의 민감도와 제약

2026-09-20 · 부모 설계 초안 · E117 완료 원자료 기반 · 실행 전 입력/자원 동결 필요

E117은 유효한 실패 진단이다. E115 선택점과 E116 두 LP 방향을 실제 비선형 조건부 응답에 대입했으나 두 방향 모두 원래99개 초기 기준 및 형태 기준에 실패했다. 두 방향 각각 최대 오차 sample이24개 qmax/LTMmax 중14개에서 바뀌었다. LM 초기 qmax는 약 .040rad에서 .142rad로 증가했다. CUDA와 부모 NumPy의6pair 응답은 일치했고, baseline의 E115 저장 손실/제약도 일치했다. 이 결과는 현재 최대 sample의 제곱오차 기울기만 제약한 선형 문제의 한계를 보여 준다. 신경 생리나 BANC 전체의 생물학적 보행 능력 부족을 증명하지 않는다.

## 다음 구현의 목적

우선 기존 저장 응답에서 **초기99개 전체 관절·LTM signed residual의 정확한 delta 미분**을 얻는 효율적인 CUDA 계산을 구현한다. 현재 최대 sample 하나에 의존하지 않고 모든 sample의 변화 방향을 다룰 수 있게 한다. 이 미분의 수치 검증이 끝나면 전체 signed sample 제약 및 원래 RMS 제약을 사용하고 실제 비선형 제약/손실로 step을 승인하는 trust-region 또는 순차 제약 최적화를 설계한다. 단순히 E115의 반복수나 E116 step 한도를 늘리지 않는다.

부모가 설계하고 같은 Sol이 구현·첫 수정을 맡는다. 부모가 두 번 리뷰하고 두 번째 리뷰 뒤 잔여 수정은 부모가 직접 수행한다. 정확한 CUDA 미분·다리/문맥/연결 축·기존 제약의 결합은 xhigh 수준이며, 별도 단순 변경을 독립 Sol에 맡길 때는 medium/high를 난이도에 따라 선택한다. 기존 에이전트 follow-up은 실제 effort를 바꾸지 않으므로 설정 변경을 주장하지 않는다.

## 저장 응답에서 계산할 민감도

모든 값은 E117 selected_solver의 두 문맥, E114의 고정 own-background, E115의 정확한 선택 delta/edge ID 순서에서 출발한다. 새 전체 BANC 또는 신체 입력 궤적을 만들지 않는다. 다른 문맥을 이어 붙이지 않는다.

각 선택 edge e=(post,pre)에 대해 W_e는 E113 실제 float32 원점을 float64로 넓힌 뒤 exp(delta_e)를 곱한 값이다. 입력 전류의 미분은 `dC_post(t)/d(delta_e) = S_pre(t) W_e`다. 기록된 C에서 `z = a(C-threshold)/cap`, `f = relu(cap*tanh(z))`이며 미분은 `1[cap*tanh(z)>0] a (1-tanh(z)^2) dC`다. 정확히0인 ReLU의 미분은 기존 PyTorch 구현과 같은0으로 둔다.

이 forcing 미분을 해당 post MN의 원래 tau를 가진 linear_response로 여과한다. 모든 muscle activation은 기존 .01초 endpoint_response를 사용하므로 이 필터를 edge별 미분에 한 번 적용한 뒤 pool/moment 계수와 곱할 수 있다. 원래0초 상태의 미분도0이다. 0..99행의 미분은 이후 시간의 forcing에 의존하지 않는다. 이 인과성을 활용하되 기존100행과2001행 FFT 구현의 roundoff 차이를 반드시 검사한다.

다리의 관절별 민감도는 `.02 * (moment @ pool)[joint,post_e]`에 위 edge별 filtered derivative를 곱한 값이다. LTM tendon은 `.2*tanh(activation_LTM/10)`이므로 미분 계수는 `.02*(1-tanh(activation_LTM/10)^2) * pool[LTM,post_e]`다. 다리별 선택 edge만 저장해 큰 구조적0 Jacobian을 만들지 않는다. 두 문맥과99행,7관절/1LTM,각 다리의 원래 selected-edge 순서를 명시한다.

## 독립 검증과 비교

새 미분 계산 전 다음 검증을 동결한다.

- 작은 CPU 예제로 analytic derivative와 실제 ContextResponse autograd를 비교한다. 양수/음수 forcing, ReLU0, 서로 다른 tau, 억제/흥분 signed weight, unsorted edge 순서와 독립 문맥을 포함한다.
- E117 저장 초기 signed residual에서 새 Jacobian을 합성해 qRMS²/qmax²/LTMRMS²/LTMmax²의8개 미분을 재구성한다. E115 선택 eval의 저장 jacobian[2:]와 모든 원래 edge에서 비교한다. peak와 동률 처리 방식도 명시한다.
- 부모는 NumPy 순차 필터로 edge별 미분을 독립 계산한다. 이것을 새로운 감각/운동 조건부 응답이나 실제 신경망 진행으로 계상하지 않으며, 미분 필터 호출·시간·출력은 별도로 기록한다.
- 저장된 E116 h를 전체 signed sample Jacobian에 곱하고 `max(abs(e+Jh))^2` 및 평균 제곱을 구한다. 현재 peak 제곱값의 affine 예측, 전체 sample 예측, E117 실제 비선형 결과를 비교한다. 새 LP나 candidate를 만들지 않는다.

원래 qRMS .02/qmax .05/LTMRMS .01/LTMmax .03 및 내부 .018/.04/.009/.025와 selector cap `(1-1e-4)^2`를 유지한다. 모든 수치 원자료는 비교 assertion보다 먼저 저장한다. 비선형 제약은 최종적으로 원래 계산 전체를 통과해야 하며, 선형 미분 검사 통과를 보행 성공이나 candidate 채택으로 취급하지 않는다.

## 실행 전에 결정할 항목

이 파일은 다음 단계의 설계안이다. 아직 E118의 실제 tensor 계산/미분 실행/LP/fit/native/신체/F6 실행을 시작하지 않았다. 구현 전에 E117 최종 부모 검토와 원자료, E115 선택 eval/Jacobian, E114 입력 및 재사용 소스의 SHA, 각 축/연결 계수, 별도 외부·내부 시간 한도와 이전 보수누적709.5144862초를 확인해 동결한다. 원래 응답을 다시 실행해 입력을 만들지 않는다. 최종 소스 리뷰 이후에만 정해진 미분 계산을 한 번 실행한다.

E118 민감도 검증 뒤의 실제 최적화는 sample별 signed 제약, RMS 제약, 작은 trust step과 실제 비선형 acceptance를 함께 설계한다. 각 시도에서 원래 모든 초기99/형태 기준을 그대로 검사하며, fixed F1–F8을 통과하는 실제 전체 BANC 후보를 얻기 전까지 제품 보행 개선 또는 자연스러운 생물학적 보행의 완료를 주장하지 않는다. 현재 마지막 실제 E113의 F3/F4/F5 실패, E110의 F3만 통과한 부분 결과, M2와 F6 미노출을 유지한다.
