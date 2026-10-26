# E205: 실제 판정 한도와 명령 민감도로 제한된 계속 학습

부모 사전 설계다. E203/E204의 실패·원자료·비용을 보존한다. E204는 40iteration을 진행했으나 명령 개선 .184002%<20%이며 수렴 종료가 아니다. 총 merit 감소 중 command 항 기여는 .081188%에 불과하고 대부분 .995 내부 여유의 벌점 감소였다. 저장된 계수별 미분의 크기도 다르다. 가설은 추가 내부 여유를 실제 판정 한도로 맞추고 command 민감도로 탐색 단위를 정하면 같은 반응식에서 더 유용한 적합을 얻는다는 것이다. 경쟁 설명은 국소 모델·보호조건·입력 문맥의 표현 한계 또는 아직 부족한 수치 탐색이다.

E89의 84위상 비음수 inverse가 RMS.001289183rad였으며 그 moment/rest/pool/membership는 현재와 값이 정확히 같다. 이 다른 분포의 낙관적 결과를 E205 성공 증거로 쓰지 않되, 같은 행렬 용량 NNLS를 반복하지 않는다. 새 세포 상태/더 풍부한 신경 함수를 도입하는 구조 대안보다 먼저 저장된 접선으로 수치 탐색을 개선한다. 고정 F1–F8은 유지한다.

## 변경과 고정

E201 단계 전류/생리/초기r,z, E202의 7기저/반응식/RK4, E150의 양수·해부학적 decoder, 48 LTM MN w0, 343MN×7 학습, w범위[-1,1], E197/E164 목표/시계/초기99를 유지한다. 전체BANC·신체 실행0, F6 미노출이다. 시작 계수는 저장된 E204 선택 평가55 하나다. 신경 시계와 r/z/접선 초기값은 매 평가마다 원래 E150 start에서 다시 시작한다. E204 마지막 r/z를 다음 평가의 초기 상태로 쓰지 않는다.

학습 목적함수의 `inner_guard`만 .995에서1로 바꾼다. 즉 전체/다리별 velocity, 다리별 command 및 startup RMS의 분모는 원래 판정 한도이며, startup 표본별 한도는 .04rad다. main20% 정규화, penalty10000, regularization1e-4*mean(w²), 모든 residual/마스크는 동일하다. 원래 .995는 E203/E204 학습의 추가 여유였고 F1–F8 또는 후보 판정 한도가 아니었다. 실제 후보 조건은 한 항목도 완화하지 않는다. 이 목적함수 변화와 변수 단위 변화가 결합된 시험이므로 각각의 인과 효과를 주장하지 않는다.

## 저장된 command 접선으로 단위 계산

E204 선택 rate_sensitivity R[t,m,k]는 초기 상태가 고정된 w에 대한 국소 rate 접선이다. 이를 E150 decoder로 전달해 각 MN이 소유한 근육 활성의 접선 D[t,m,k]를 계산한다. 초기 bank/pooled activation의 접선은0이다. 각 5/10/40ms EMA에 R[1:]를 통과시키고, 실제 저장 bank>kappa이면 high slope, 아니면 low slope를 곱해 합친다. 가변77근육은 이 값을 사용한다. fixed19는 원래 pool[group,m]×10ms EMA(R)를 사용한다. pool의 비소유 항목은0임을 확인한다. row0 접선은0이다. 이 계산은 새 신경/물리/조건부 replay가 아니다.

각 열(m,k)의 command Jacobian은 J[t,j,m,k]=.02*moment[j,membership[m]]*D[t,m,k]다. 실제 target_indices의1824행에서 전체 명령 항의 Gauss–Newton 대각과 regularization만 계산한다:

`h[m,k] = 2*sum_j((.02*moment[j,group])²)*sum_target_t(D[t,m,k]²)/(1824*42*(.8*original_command_RMS)²) + 2e-4/2401`.

이는 신경/decoder의 2차 미분이나 모든 벌점을 포함한 진짜 Hessian이 아니다. 반응 없는 열도 regularization으로 양수 h를 가진다. free343×7 순서로 펴서 `exponent=clip(floor(-.5*log2(h)),-30,0)`, `S=2**exponent`로 고정한다. clip은 변수 단위에만 적용하며 w를 자르지 않는다. 2의 거듭제곱 S를 사용해 시작 계수와 bounds의 float64 왕복이 exact인지 확인한다.

`w=S*u`, `gradient_u=S*gradient_w`, `u bounds=±1/S`, `u_initial=w_E204/S`다. S는 첫 fit 전에 원자료와 함께 저장하고 fit 중 갱신하지 않는다. initializer의 w 변환과 E204 원본 weights/free_weights는 exact여야 한다. 이를 만족하지 않으면 optimizer0으로 종료한다.

## 실행·검증 계약

실제 gpt-6-sol/high가 새 `tools/fit_c_banc_m3_command_conditioned_response.py`, `tests/test_c_banc_command_conditioned_response.py`만 구현한다. E203/E204/E202 소스는 변경하지 않는다. E203 PersistentLocalResponse/DecoderAdjoint/validate_response/direct_metrics/feasible/load_targets, E202 loader/IO를 읽기 전용 재사용한다. 새 objective helper는 원래 식의 inner_guard1 버전이다. 새 독립 runner/source gate이며 기존 gate를 monkeypatch하지 않는다. 부모 FIRST→필요시 동일Sol 첫수정→부모 SECOND/잔여수정이다.

생성자료에서 decoder의 D를 직접 유한차분으로 검산한다(가변/fixed/LTM, 같은 muscle의 복수MN, row0 고정 포함). J 열의 제곱합으로 h를 독립 검산하고 scale의 power-of-two/bounds/왕복과 chain rule을 확인한다. actual-limit objective는 활성 norm/sample 벌점의 central FD로 확인한다. abs2e-5+rel2e-4를 쓰되 kink를 피한다. 동일 w cache/45회상한/기준복원 실패 및 비유한 신경 배열/오류 후 원자료 보존을 확인한다. 변경 없는 CUDA 커널 시험은 반복하지 않는다.

원본 첫 평가(45회에 포함)는 E204 선택 weights의 실제 CUDA 재계산이다. E204 네 신경 배열 bit exact, decoder atol2e-12/rtol1e-12, 기존 E203 목적함수로 계산한 merit/gradient_w와 E204 평가55의 값 atol2e-12/rtol1e-12를 확인한다. 그 후 새 목적함수로 한 번 학습한다. 기준복원 실패면 optimizer0, 원자료 저장, 자동재시도0이다.

SciPy1.18.1 L-BFGS-B1회, jac=True, maxiter40/maxfun45/maxls30/maxcor10/ftol1e-12, gtol=min(S)*1e-6이다. 이는 native gradient의1e-6보다 느슨하지 않은 scaled gradient 조건이다. 실제 CUDA≤45(초기 포함), w bytes cache, random/restart0다. 전역 최적성 또는 모델 전체 불가능성을 주장하지 않는다.

판정은 그대로 전체 command RMS≤원래 E150*.8, 전체/각다리velocity와 각다리command 비악화, startupq RMS≤.018/max≤.04, startupLTM RMS≤.009/max≤.025, 계수/해부학/48LTM 보존이다. feasible 중 command RMS 최소, 없으면 merit 최소, 동률 최초를 선택한다. main/보호조건의 raw 지표로 직접 판정하며 solver 성공 여부만으로 합격하지 않는다. 유한한 최대반복/평가/선탐색 종료는 bounded search 결과, 비유한/계산/복원/시간/출력 실패는 invalid 또는 inconclusive다. 조건부 성공과 실제 보행·F1–F8 완료를 구별한다.

preconditioner.npz에 D, h/scale/exponent/free_columns/initial_weights를 저장한다. 평가별 weights/free_weights/scaled_variables/gradient/scaled_gradient/command와 metrics/terms/merit/순서/시간을 다음 평가 전에 보존한다. 선택한 네 신경 원자료/decoder/weights/motor IDs/u/scale을 최종 판정 전에 저장한다. 실패 시 확보한 최선 자료도 unverified_incomplete_raw로 보존하고 원래 오류를 유지한다.

원본process1/optimizer1/실제CUDA≤45/CPU1/GPU1/추가warmup0/retry0, fit120/내부135/외부145/감독150초, 출력256MiB/GPUbuffer768MiB/free disk4GiB다. 누적5285.855097999705/5444초·잔여158.14490200029468초에서150초를 예약하며 배정을 늘리지 않는다. 최대 조건부450초/180만tick/720만stage다. 준비/생성/부모 검산은 별도이며 미측정은0이 아니다. 원래 whole goal은 active·미완료, M2/E150/UI/과거 실패/F6 보존이다. 결과 통과 후 실제 BANC/신체 시험을 설계하며, 미달은 저장된 최적화·표현/보호조건 근거로 재판단한다.
