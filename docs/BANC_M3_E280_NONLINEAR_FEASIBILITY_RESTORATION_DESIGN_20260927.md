# E280 — 비선형 시작 제약의 보정 단계

2026-09-27 root 설계. F1–F8, E277 train/LM 미달, E278 기술 실패, E279 실제 시작 max .040869485 > .04 기각을 보존한다. E279의 초기 조건부–실제 차이는 작지만 학습 안전 여유가 부족했다. 본 단계는 실제 보행 승격이나 E277 재시도가 아니라 **학습 제안 거절 원인의 제한 진단과 재사용 가능한 보정 구현**이다.

## 근거와 바뀌는 결정

root와 실제 gpt-6-sol/xhigh가 E277 저장 history/direction을 별도로 읽었다. 167 거절은 모두 실제 조건부 startup 위반이다(q peak 166, q SSE 12 중복, LTM 0). Armijo 실패 5개도 이 집합에 포함되고 Armijo만 실패한 것은 0이다. 마지막 6개 방향은 모두 인증됐고, gradient max .498–1.973과 음의 방향 미분이 남아 있었지만 radius는 1e-6–1.2207e-5였다. 전부 24 outer에 도달했고 시간/80평가 한도는 아니었다. 원래 소스 필터 가족이 불가능하거나 수렴 완료했다고 단정할 수 없다.

최초 full step의 선형 q peak는 .03938–.03989였지만 실제 조건부 peak는 .04265–.08473였다. 처음에는 비선형 오차가 크고, 마지막 1e-12–1e-9 위반은 수치 잔여와 비선형 효과를 구분해야 한다. 보정으로 더 큰 **엄격히 유효한** 이동을 얻는다면 다음 학습 방법에 반영할 근거가 된다. 보정해도 같은 제약을 만족하지 못하면 반복 확대나 수용 오차 완화 없이 이 한정된 방법을 기각한다. 현재 source 가족의 전체 불가능을 뜻하지 않는다.

## 하나의 고정 비교

각 LF/LM/LH/RF/RM/RH에서 E277 outer 0의 마지막 rejected 제안 한 개를 사용한다. 선택된 alpha는 .25/.5/1/.25/.03125/.5이고 evaluation은 3/2/1/3/6/2다. 같은 outer에 채택한 작은 이동을 비교한다: .125/.25/.5/.125/없음/.25. RM 비교는 당시 유지한 중립점이다. 결과를 본 뒤 제안 또는 비교군을 바꾸지 않는다. 초기 점·거절 점·비교점·원래 fixed_sse·목적 정규화·구성·목표를 모두 SHA로 고정한다.

먼저 같은 E273 exact 조건부 엔진에서 이 여섯 거절점과 각 before 점을 다시 계산하여 E277 저장 q/LTM/objective와 before gradient를 재현한다. 원래 손실/모델의 미소 수치 차이는 각각 q/LTM 1e-10, objective 1e-10, before gradient 1e-9 절대 상한으로 판정하고 차이 원인을 숨기지 않는다. before 및 trial 평가를 cache에 보존한다. 재현 실패면 해당 보정은 실행하지 않고 원본 전체를 invalid로 남긴다. 기존 E268/270/273/E277 동결 자료만 읽고 미사용 F6나 개발 목표를 학습에 사용하지 않는다. **실행 전 자료 확인 정정:** 거절 trial gradient는 기존 eval 파일에 저장돼 있지 않다. 그 값은 새 평가 산출물로 보존하며 과거 원자료와의 재현을 주장하지 않는다.

실행 순서는 LF부터 **case별 before/trial 재현 → 최대 두 보정 → 두 엔진 해제**다. 다음 case 재현 실패 시 이전 부분 자료·비용은 보존하고 전체는 invalid로 판정하며 후속 case는 실행하지 않는다. 12 엔진/max 2 live 한도를 지키기 위해 여섯 재현을 동시에 보관하거나 엔진을 다시 생성하지 않는다.

거절점에서 얻은 **새 실제 Jacobian**으로 최소 norm 보정을 최대 두 번 수행한다. 여기서 새 Jacobian은 고정 입력 조건부 신경 방정식의 미분이며 전체 BANC 재귀 미분이 아니다. 이전 선형 예측만 다시 사용하지 않는다. correction 결과를 실제 비선형 엔진으로 평가하고 유효하지 않으면 그 현재 점에서 한 번 더 보정한다. 자동 line search/추가 alpha/재시작/세 번째 보정은 없다.

## 보정 수학과 수용

`z = correction`, objective `.5*z.T*z`인 SOCP를 풀어 다음을 만족시킨다. 원래 before 계수에서 반경 .05인 상자와 원래 coefficient [0,8]/triple 합 [ .125,8 ]의 **기존 interior**를 유지한다. 수정량 자체는 선택된 원래 거절 이동의 최대 절대값 이하로 제한한다. 누적 두 번 보정도 원래 before trust box를 벗어날 수 없다. clip/projection으로 해를 사후 변경하지 않는다.

`e_trial + J_trial*z`의 모든 99×7 q와 99×1 LTM peak를 원래 내부 peak(.04/.025)*(1-1e-4) 이하로, norm을 `sqrt(N_global*(RMS*(1-1e-4))**2 - fixed_sse)` 이하로 둔다. N_global은 q 99×42, LTM 99×6이며 fixed_sse는 해당 E277 원래 값이다. 내부 선형화 한도에만 1e-6의 절대 여유를 빼서 1e-8 solver primal 인증 오차보다 100배 크게 둔다. 이는 실제 수용 기준 완화가 아니며 E279의 전체망 오차에 대한 보장도 아니다. 음수 남은 norm 예산은 실패다. 현재 trial이 불가능해도 보정 문제를 작성할 수 있어야 한다.

Clarabel 한 호출당 max_iter 100/time_limit 2초/threads 1/tol 1e-8. P=I를 canonical sparse로 전달한다. 결과는 Solved/AlmostSolved만 독립 인증 대상으로 삼는다. exact slack `s=b-Az`의 모든 cone >= -1e-8, 원래 coefficient/trust/linear/SOC residual >= -1e-8, dual cone >= 0 (clip/허용오차 없음), stationarity `z+A.T*y` infinity <=1e-8, primal `.5*z.T*z`와 dual `-.5*||A.T*y||²-b.T*y`의 gap [-1e-10,1e-8]*(1+max(abs(primal),abs(dual))), complementarity 동일 1e-8 상대 상한을 함께 검사한다. raw solver slack/dual/status/iterations/failed calls를 원형 보존한다.

실제 보정 점은 기존 `LegEvaluator`의 엄격한 원래 startup 및 coefficient 검사, train 비악화, before→corrected 음의 slope와 기존 Armijo를 모두 통과해야 한다. 솔버 잔차 허용치를 실제 조건부 수용에 사용하지 않는다. 기준을 만족하면 즉시 멈춘다. 2회 후 미달은 reject, 저장·계산 오류나 deadline은 invalid/incomplete다. 개선 비교는 기존 채택점보다 실제 objective가 낮은지로 별도 기록한다. 비교점보다 못하면 보정 성공을 학습 개선으로 부르지 않는다. 최종 f32 평가/전신 평가를 하지 않으며 이 결과에서 후보를 만들거나 선택하지 않는다.

## 구현과 검토

새 `tools/banc_m3_feasibility_restoration.py`, `tests/test_c_banc_feasibility_restoration.py`, `tools/verify_c_banc_m3_feasibility_restoration.py`, `tests/test_c_banc_feasibility_restoration_runner.py`만 수정한다. 기존 E270/E273/E275/E277/E279 소스·M2/UI는 그대로 둔다. 실제 gpt-6-sol/xhigh가 구현한다(신경 미분, 비선형 제약/독립 dual 인증, 실패 비용의 연결 때문). Root FIRST → 필요하면 같은 Sol 첫 수정 → root SECOND → root 잔여 수정이다. Sol은 작은 생성 CPU 문제/mock만 실행하고 원본/대형 solver/CUDA/body/새 하위 위임은 0이다.

Helper는 일반 배열/콜백을 통한 원래 evaluator와 연결하고 import 중 실행하지 않는다. Runner는 freeze와 source review의 해시/소스/패키지/자료/고정 6개 case/내부 한도를 검증한 뒤에만 엔진을 생성한다. E277 freeze에서 mapping/target을 명시적으로 로드·다리별 축소하고 teacher는 1..1200 학습만 노출한다. max 2 context live, 각 다리 startup101/teacher1201. Root가 준비하는 case NPZ에 before/trial/comparator c, saved q/LTM/objective/gradient, fixed_sse, baseline_mse, 원래 radius와 각 의미를 저장하고 old direction과 비교한다. no implicit leg slicing; 기존 명시 index roster 사용. 모든 실제 단계/계수/예측·원자료·선형 문제·raw solve·판정/부분 비용/예외/cleanup을 저장한다. 기록 실패는 성공이 될 수 없다.

생성 검증은 가능한 최소 norm 해의 analytic 비교, 불가능 문제, 시작점 infeasible 허용, original box/triple 유지, 모든 peak/SOC/global fixed SSE, dual/gap/status 위조 거부, 실제 제약이 여전히 깨지면 거절, 두 보정 상한/조기 성공/원래 Armijo, 재현 실패에서 solver0, 저장 오류·원래 exception·cleanup·예산 중단 보존을 포함한다. root는 소스를 검토하고 별도 생성 곡선 사례와 저장행/독립 산술로 비교한다.

## 자원과 다음 경계

현재 원본 7823.27558739961/7905초, 잔여81.72441260038977초. 원본은 한 번, supervisor70초(과학45+정리25), RAM6GiB/GPU추가4GiB/output2GiB/free disk4GiB, CPU1/동시CUDA1, 최대12조건부 엔진·12 SOCP·24 coefficient evaluations다. 각 평가는 .5초+6초 조건부 26,000 forward/derivative ticks이므로 각 최대624,000 ticks다. 전체 BANC/body/물리/새 fit/full development/F6는0이다. 준비·생성검사·원본·수동리뷰를 구분하고 실패 비용을 누적한다. Source 승인 전 원본 예약0이다.

이 비교가 지원할 수 있는 것은 보정 단계가 기존 줄인 이동보다 유효한 개선을 주는지까지다. 여섯 다리 모두 재현·엄격 보정·비교점 개선을 통과할 때만 다음 full-fit 설계의 후보 방법으로 삼는다. 일부 성공은 다리별 기록만 남기고 전신 학습 승격은 하지 않는다. E279의 native 여유 및 최종 학습20%/개별 개발/F1–F8은 다음 full-fit 설계에서 별도로 해결해야 한다. 원래 제한을 지키며 결과를 얻지 못하면 불확실성을 명시한다.
