# E203: 해부학적 전달을 유지한 국소 MN 반응 학습

부모 설계, 원본 실행 전 계약. E202의10초 원본 복원과 독립 검산은 통과했다. 초안의 유한 목표 F1–F8과 과거 실패를 유지하며 이번 계약을 실행한다. 신경 발화 후 근육 전달만 바꾼 가족의 실패와 달리, 기존 r/z와 입력으로 결정되는 세포 반응을 학습한다. 기록된 전류에 대한 조건부 적합이므로 전체 회로의 되먹임이나 실제 보행 개선을 입증하지 않는다.

## 입력·개입·비교

E201의 실제160000×391 RK stage current, 초기r/z, 실제 float32 생리 계수를 사용한다. E202의7기저·0.5보정폭·[-1,1]계수와 float64 RK4 접선식 그대로다. 초기 상태/threshold/coefficient/gain/inverse tau/연결/decoder는 고정한다. E150의 해부학적 MN→96근육 소유권과 기존 단조 모집층을 유지한다. 학습은 MN의 국소 반응 w만 바꾼다.

여섯 LTM의48 MN은 w=0으로 고정한다. 나머지343 MN×7=2401계수를 graph column 오름차순, basis0..6 순으로 학습한다. singleton의 신경 반응은 포함하되 원래 근육 전달은 그대로다. 무출력 MN을 강제 활성화하지 않는다. mask는 실제 root/graph ID/근육 소유권에서 동결한다. 해부학적 연결 추가, signed 근육 전달, runtime 외부 위상·시계·참조 궤적·root 보정은 없다.

목표는 E197 pre-solve raw에 보존된 E85 own-leg/offset 참조, E103 초기99이다. target_indices100..1923의1824행, 인접velocity1823쌍, startup1..99와 six tendon 목표를 유지한다. 해당 원자료를 선택적으로 읽어 현재 E150 기록과 다시 대조한다. baseline은 원래 recorded_command이며 새 계산기의 결과로 바꾸지 않는다. 초기 row0는 held command를 보존하고 적합에 쓰지 않는다. F6는 미노출이다.

## 계산과 미분

E202의 검산된 CUDA_SOURCE, float64/--fmad=false 옵션, 128thread block을 그대로 재사용한다. 새 장치 객체는 current/실제 생리/출력 buffer를 한 번 할당한다. 각 평가에서 w만 전송하고 초기r/z/접선0에서 동일10초를 계산한다. 원본 계산마다 신경 state를 이어붙이지 않는다. 입력을 추가 warmup에 쓰지 않는다.

첫 w=0 평가는 실제 평가60회 한도에 포함한다. rate/adaptation 및 두 접선 전체를 E202 raw와 bit exact 대조하고, decoder 출력은 max2e-12+rtol1e-12로 비교한다. 불일치면 이후 optimizer를 실행하지 않는다. 이 검증을 통과한 w0의 목적함수·gradient를 cache하여 optimizer가 같은w를 요청하면 중복 CUDA를 하지 않는다.

GPU의 원래391MN 결과에 E150의 causal EMA 5/10/40ms와 original pooled10ms를 적용한다. SciPy lfilter 또는 동등한 vectorized 계산을 허용하며 생성자료 및 원본w0에서 원래 decoder와 검산한다. 가변77근육에는 original low/high slope·kappa를, fixed19에는 원래 pooled activation을 쓴다. q=rest+.02*moment@activation과 tendon=.2*tanh(LTM/10)은 그대로다.

미분은 q gradient→.02 moment 전치→근육 gradient→해부학 소유권별 slope→EMA adjoint→rate gradient로 계산한다. bank=κ에서 hinge 미분은0으로 정의한다. fixed19는 원래 pooled10ms EMA로 미분한다. 시간축 rate gradient와 국소 rate_sensitivity를 수축하여 MN×7 gradient를 만든다. 초기 상태는 학습하지 않으며 row0 입력에 대한 gradient를 전달하지 않는다. 거대한 전체 command Jacobian을 만들지 않는다. 다른 MN의 재귀 전류 변화는 포함하지 않는다.

## 한 번의 제한된 탐색

SciPy1.18.1 L-BFGS-B를 한 번 사용한다. 초기w=0, bounds[-1,1], jac=True, maxiter40/maxfun60/maxls10/maxcor10, ftol1e-12/gtol1e-6이다. random initialization/restart0이다. 전체 실제 CUDA 평가를60회로 제한한다. 동일w의 정확한 byte key에 대한 cache는 중복 계산하지 않으며 cache hit도 기록한다. 최초w0 평가 포함이며 optimizer 내부의 추가 평가가61회가 되기 전에 중단한다. 이는 정상적인 bounded search 종료로 기록하고 전역 최적성이나 가족 반증을 주장하지 않는다.

Eq=q[target_indices]-target, Ev=(q[pair_end]-q[pair_start])/.005-target_velocity, Es=q[1:100]-startup_target이다. L0=mean(Eq²)/(.8*원래전체commandRMS)²로 둔다. 보호 항목14개는 전체velocity, 각6다리command/velocity, startupRMS이다. 각 항목의 Aj는 해당 residual MSE를 (.995*원래한도)²로 나눈 값이다. startupRMS의 원래한도는.018rad, 다른 항목은 원래 baseline RMS다. Lj=10000*max(Aj-1,0)²를 합산한다. 추가 startup 표본별 B=(Es/(.995*.04))²에10000*sum(max(B-1,0)²)를 더한다. 마지막으로1e-4*mean(free_w²)를 더한다. 모든 항의 실제 값과 해석적 gradient를 기록한다. 이 벌점의 내부 여유는 판정 기준을 완화하지 않는다.

판정은 원래 전체commandRMS의.8 이하, 전체/다리별velocity와 각다리command 비악화, startupq RMS.018/max.04, startupLTM RMS.009/max.025다. 원래48 LTM MN의 w0와 실제 생리·decoder 보존, 모든 계수의 유한값/범위가 필요하다. 비교에 임의 여유를 추가하지 않는다. 실제 모든 조건을 통과한 평가 중 commandRMS가 가장 작은 평가 하나를 선택한다. 없으면 merit가 가장 작은 평가를 미달 진단으로 선택한다. 동률이면 먼저 수행한 평가다. clipping·정규화·후보 조합·추가 최종 재실행은 없다.

최대평가/최대iteration/line-search 종료라도 실제 평가와 반환이 유한하고 기록되었다면 제한된 탐색 결과로 판정한다. 비유한값, 계산·기준복원·시간/출력 한도 실패는 invalid/inconclusive로 분리한다. 통과 시 supported_conditional_local_response_proposal, 미달 시 valid_bounded_search_no_proposal이다. 둘 다 채택이나 F1–F8 완료가 아니다.

## 저장·검증·자원

평가별 w/gradient/command/직접지표/merit 및 순서·시간·cache 여부를 다음 평가 전에 새 파일로 보존한다. 실제 선택 규칙에 따라 현재 최선의 rate/adaptation/R/Z는 CPU 메모리에 보관하고, 마지막에 선택한 원자료·bank/activation/command/tendon을 저장한 후 최종 판정한다. 전체 평가의 w와 명령을 버리지 않는다. 반환 실패 뒤 원본을 자동 재실행하지 않는다. 상태 전개 횟수와 GPU launch 요청/동기화 완료/불명확 횟수를 구분한다.

actual gpt-6-sol/high가 새 tools/fit_c_banc_m3_local_response.py와 tests/test_c_banc_local_response_fit.py만 구현한다. E202 및 기존 소스는 읽기 전용으로 재사용한다. 부모 FIRST→필요시 같은 Sol 첫수정→부모 SECOND/잔여수정이다. 생성자료의 variable/fixed/LTM 전달, EMA adjoint, loss gradient와 국소 접선의 결합을 central FD 및 E202 NumPy로 검산한다. abs2e-5+rel2e-4를 사용하되 kink를 피한다. 중복cache/60회상한/실패 후 raw 보존도 검사한다. 원본은 부모 source gate 이후 한 번이다.

원본 process1/optimizer1/CUDA평가최대60/자동retry0, CPU1/GPU1이다. fit240/내부270/외부285/감독300초, 출력256MiB/GPU buffer768MiB/free disk4GiB다. 실제 전체 BANC 신경·신체0이며 조건부 최대600초/240만tick/960만stage다. 기존 누적5112.448676599703/5444초, 잔여331.5513234002974초에서300초를 예약하고 배정을 늘리지 않는다. 준비·합성·부모 검산 비용은 별도 기록하며 미측정을0으로 쓰지 않는다.

조건부 통과 시 같은 방정식을 실제 전체 BANC stage에 적용하여 outgoing feedback·기존 감각·실제 신체·저장/복원을 검증하는 짧은 후속시험을 설계한다. 미달이면 고정 입력에 대한 실제 잔차/보호조건/gradient를 확인한다. M2/UI/최선E150와 과거 실패를 보존한다. 자연 보행과 생물학적 동등성은 아직 미검증이다.
