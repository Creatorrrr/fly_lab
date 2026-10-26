# E119: 모든 초기 sample을 제약하는 두 문맥 가중치 적합

2026-09-20 · 부모 설계 · Sol/xhigh 구현 및 첫 수정 · 부모 재리뷰와 잔여 직접 수정

## 문제와 고정 범위

E115는 초기 최대 제곱오차의 현재 peak 기울기만 사용한 SLSQP에서 형태 개선에 실패했다. E117은 저장된 두 LP 방향 모두 실제 초기 제약을 위반함을 확인했고, E118의 모든 signed sample 미분은 그 위반을 포착했다. 두 방향에서 초기 ratio 예측 RMS 오차가 3.338732→.073218 및 2.889204→.057859로 줄었다. E119는 이 검증된 제약 표현을 사용한 새 최적화다. 기존 E115 반복 수만 늘리는 재시도가 아니다.

목표는 원래 두 기록 문맥 E110/E113에서 초기 자세 제약을 지키면서 동일한 관절·발·발속도·LTM 목표 오차를 줄일 수 있는지를 확인하는 것이다. 전체 F1–F8, 실제 E113의 F3/F4/F5 실패, E110의 F3 부분 결과, M2 원본과 F6 미노출을 유지한다. 조건부 proxy 적합의 성공을 실제 신경망 보행이나 생물학적 기전 검증으로 간주하지 않는다.

E114 fixture, E115 LegStage/기존 손실/ID 기반 float32 export/판정, E111 geometry, E118 derivative_bank를 수정 없이 재사용한다. 원래 391 MN/49 source/96 muscle/42 joint/6 LTM/1572 selected edges, 두 문맥의 고정 background, 생리 매개변수, 초기 zero state와 .005초/2001행을 유지한다. graph/node/edge/sign/근육 연결 또는 감각·물리 모델은 바꾸지 않는다. 초기99행(1..99), 공통1823행, fixed1828행과 own-phase/legacy 판정도 불변이다.

## 변수·손실·제약

다리 순서는 LF, LM, LH, RF, RM, RH다. 각 다리의 원래 edge subset에 delta와 t를 둔다. 후보 W는 E113 실제 native float32를 float64로 확장한 기준 W에 exp(delta)를 곱한다. delta bounds는 E115 trust_bounds가 정한 ±log(1.1)과 원래 efficacy [.001,10]의 교집합이다. 범위를 확대하거나 새 step을 clipping해서 다른 후보로 바꾸지 않는다.

각 다리의 유일한 시작점은 E115 selected-delta의 해당 subset이다. 이 시작점도 첫 unique delta 응답으로 계상한다. 초기 t는 max(two_context_losses)+.01이다. E115와 같은 네 정규화 손실(q/foot/velocity/LTM, 동일 가중치)을 합한 문맥별 loss를 사용한다. 목적은 t + 6e-5*sum(delta^2)/1572, 제약은 두 문맥 각각 t-loss >= 0이다. optimizer cap=(1-2e-4)^2, selection cap=(1-1e-4)^2를 그대로 유지한다.

초기 signed error를 q(2,99,7), z(2,99)라 한다. inner caps는 [qRMS .018, qmax .04, LTMRMS .009, LTMmax .025]다. 부등식 순서는 다음과 같이 고정한다(C order flatten, context/time/joint 순서).

1. t-loss[0:2]: 2행.
2. sqrt(optimizer_cap)-q/.04: 1386행.
3. sqrt(optimizer_cap)+q/.04: 1386행.
4. sqrt(optimizer_cap)-z/.025: 198행.
5. sqrt(optimizer_cap)+z/.025: 198행.
6. optimizer_cap-mean(q^2)/.018^2: 문맥별 2행.
7. optimizer_cap-mean(z^2)/.009^2: 문맥별 2행.

합계3174행이며 모든 부등식은 >=0이다. signed 최대 제약은 원래 최대 제곱오차 제약과 수학적으로 같은 feasible set을 표현한다. 현재 peak만 고르는 비미분성을 제거하며 기준을 완화하지 않는다. RMS gradient는 모든 sample의 실제 signed error와 Jacobian을 평균한 정확 미분이다. 원래 [qRMS .02,qmax .05,LTMRMS .01,LTMmax .03] 보고/채택 기준은 그대로 남긴다.

SciPy SLSQP 한 번/다리, exact objective·constraint Jacobian, maxiter100/ftol1e-9, 재시작0이다. 매 평가에서 원래 비선형 응답과 gradient를 새로 계산하고 SLSQP line search를 사용한다. 반환 success만으로 채택하지 않는다. 실제 저장된 8개 초기 squared ratios가 selection cap 이하인 점 중 true worst loss+prior가 가장 작은 점을 선택한다. 실패점이나 affine feasibility는 선택 근거가 아니다. 초기 warm start도 같은 검사를 통과해야만 선택 가능하다. 한 다리에 feasible 점이 없으면 모든 다리 기록을 보존하고 no_joint_feasible_candidate로 끝내며 전체 후보 응답을 실행하지 않는다.

## CUDA 응답과 미분, 원자료

각 unique delta에서 LegStage.tensors를 정확히 한 번 호출한다(두 문맥, 각10조건부초). 두 scalar loss의 exact gradient만 autograd로 구한다. 초기 sample gradient는 방금 계산한 response.current/activation의 첫100행과 같은 후보 W를 사용한다. E118 derivative_bank의 generic local MN/edge API를 torch.no_grad로 호출한다. 원래 forcing의 ReLU0 derivative=0, post별 tau와 .01초 muscle filter, signed W를 유지한다. 추가 forward를 호출하지 않는다. linear/endpoint derivative filter는 각1회/unique delta이며 forward filter와 별도 계상한다.

bank의 filtered edge derivative v(2,100,nEdge)에서 Jq=.02*(moment@pool)[joint,post_edge]*v[:,1:]이다. JLTM=.02*pool[LTM,post_edge]*(1-tanh(actual_activation_LTM/10)^2)*v[:,1:]이다. local muscle/MN/edge 순서는 명시적 ID와 index로 검증한다. E118 full391 leg_jacobian을 local shape에 잘못 재사용하지 않는다.

저장 공간과 callback 비용을 줄이기 위해 매 unique delta에는 v, q/LTM signed errors, 두 losses/네 parts/8 ratios, 두 loss gradient, 해당 delta/edge IDs, q coefficient(7,nEdge), LTM coefficient(nEdge), LTM gain(2,99)을 저장한다. 이것으로 모든 Jq/JLTM과3174행 부등식/Jacobian을 정확히 재구성할 수 있다. 매번 full signed Jacobian의 양·음 복사본을 저장할 필요는 없다. np.savez의 비압축 compact raw를 허용한다. raw를 assertion 전에 저장한다. 전체 forward time traces와 bank intermediate는 warm start 6개 및 실패 시에 보존한다. 일반 평가의 loss/gradient/모든 signed residual/factored Jacobian은 모두 남긴다.

delta의 float64 bytes만 캐시 키로 사용한다. t만 바뀌거나 이미 평가한 delta로 되돌아오면 응답/미분 bank를 다시 호출하지 않는다. 캐시는 compact factor만 유지하고 constraint algebra는 NumPy에서 재구성해도 된다. CUDA는 신경 응답·기하/손실 미분·초기 derivative filter에 사용한다. CPU SLSQP와 작은 제약 배열 조립을 CUDA 계산으로 표기하지 않는다.

첫 평가에서 저장 raw 후 E115 selected eval과 parts/loss/ratios 및 두 loss gradients 최대차1e-8을 검사한다. 해당 E118 jacobian-{leg}.npz와 q/LTM errors/Jq/JLTM 최대차1e-9도 검사한다. 이것은 각 다리 첫 응답에 포함되며 별도 baseline response나 derivative bank가 아니다. 실패 시 새 초기화/추가 fit 없이 원본을 보존하고 종료한다.

모든 입력 전송 완료 후 durable start receipt를 성공시킨 다음 실제 호출한다. 시작/함수 진입/반환·materialized/raw saved를 구분하고 응답 pair/조건부초와 derivative bank/filter 수를 별도 기록한다. 준비 또는 시작 receipt 실패 시 실제 호출0이다. 응답/미분/gradient/후속 저장 실패 시 이미 알려진 수치·delta·ID·weights·시간축을 복구하고 원래 예외를 유지한다. secondary 저장 오류는 원래 예외를 덮지 않는다. raw 저장 전에 finite assertion을 실행하지 않는다. 재사용 helper의 입력 finite guard는 허용한다.

## 최종 후보와 검증

여섯 다리에 feasible 선택점이 있으면 delta를 결합하고 E115의 원래 efficacy 경계 roundoff 검사와 명시적 ID 기반 selected_actual_float32를 사용해 calibration과 실제 float32 W를 진단 출력으로 만든다. unselected W, topology/sign/기존 calibration의 다른 필드는 불변이어야 한다. solver W와 실제 source*efficacy float32 W 차이를 숨기지 않는다. 경계 허용은 기존8ULP 원래 연산 반올림에만 적용하며 delta bounds는 확대하지 않는다.

최종 float32 W에 대해 전체391 MN/두 문맥10초 응답을 한 batch로 딱1회 실행한다. E115 frozen_output_metrics로 원래 모든 screen을 계산한다: 양 문맥의 common foot 및 velocity 각각 E113 기준20%이상 개선, q 악화5%이내, 원래99/내부 초기 판정, fixed/own-phase/legacy 지표. true robust_local_supported는 기존 함수 판정을 따른다. 채택/실제 신체 실행은 별도 부모 리뷰 후 다음 단계에서 결정하며 이 runner는 candidate_adopted=false/native_gait_tested=false/F6_exposed=false를 유지한다.

부모는 전체 후보 raw가 있을 때만 독립 NumPy 두 문맥 응답을 딱1회 계산한다(내부15/외부20초). 부모의 기존 독립 코드와 raw 지표 검산으로 결과·원래 bounds·선택·초기전체 제약·float32 export·모든 원본 hash·자원을 확인한다. 평가 중 delta마다 부모 응답을 반복하지 않는다. warm-start gradient는 E118의 이미 독립 검증된 자료를 사용한다.

## 실행 한도와 순서

새 tools/fit_c_banc_m3_samplewise_startup.py와 tests/test_c_banc_samplewise_fit.py만 Sol이 구현한다. 기존 source/설계/원자료/제품/UI는 수정하지 않는다. 결합된 신경 미분·CUDA·다중 제약·한도/예외 처리가 있어 기존 Sol gpt-5.6-sol/xhigh를 유지한다. 부모 첫 리뷰 뒤 첫 수정은 같은 Sol, 두 번째 리뷰 뒤 남은 수정은 부모다.

합성 CPU 테스트는 작은 실제 ContextResponse의 전체 signed constraint autograd/Jacobian 일치, 두 문맥·억제/흥분·다른 tau·ReLU0, signed sample과 squared max feasible set 동치, peak가 바뀌는 경우, t-only/backtracking cache, 범위/selection, raw 구조와 실패 복구/회계, 작은 SLSQP 한도와 no-feasible 반환을 다룬다. 최종 소스 리뷰 전 원래 dataset forward/derivative/fit은0이다.

승인된 원래 fit은 RTX4080 cuda:0/Torch float64, OMP/MKL/OPENBLAS/Torch CPU thread1에서 전체 한 프로세스만 실행한다. 최대100 unique delta/다리, 총600개, 1200단일다리 문맥 pair/12000조건부초다. derivative bank최대600개/각 filter최대600회, SLSQP최대6회다. warm start는 이 한도에 포함된다. cache hit은 새로운 평가가 아니다. final 전체응답최대2pair/20조건부초는 별도 계상한다. 실제 full BANC/native/body/F6=0이다.

다리 wall24초(21초부터 다음 평가를 시작하지 않고3초 정리 여유), fit150초, 내부 process175초, 외부200초(child timeout195초)다. soft stop은 저장된 점 선택 후 다음 다리로 진행하며 다리/fit/process hard timeout은 전체 종료한다. solver 한 번의 긴 callback 사이에도 외부 제한을 적용한다. 각 다리 최종 선택/저장까지 다리시간으로 포함하고 fit 최종 deadline 검사 후에만 fit clock을 닫는다. E118 완료 누적716.4107936초를717로 올려 별도 shared ceiling917초를 사용한다. 과거 E115838/E116730/E117731/E118735초 한도와 실패를 소급 변경하지 않는다. 출력 e119-samplewise-fit/는 이미 있으면 거부, 재시도0이다.

외부 stdout/stderr/exit/timeout/프로세스 wall/원본 manifest와 parent review를 남긴다. 숫자 미달은 유효한 실패로 기록한다. 반복 수·범위·기준을 늘리지 않고 새 evidence에 근거해 다음 단계를 설계한다. F1–F8 전체 goal은 계속 active이며 단순 진단/fit/test 성공으로 완료 처리하지 않는다.
