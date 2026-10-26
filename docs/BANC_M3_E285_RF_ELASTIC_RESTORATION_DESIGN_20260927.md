# E285 — 실제 peak 감소를 따르는 RF 비선형 시작 보정

2026-09-27 root 사전 설계. E284 중립의 full-domain 고정 J는 최종 peak를 만족할 수 없지만 E283 실제 중간 점에서는 J가 달라졌다. E281–E284의 엄격 선형 feasibility 요구는 실제 비선형 영역으로 이동하기 전에 중단될 수 있다. H285는 출력 slack을 중간 선형 문제에만 도입하고 실제 개선을 확인하면 같은 neural/decoder/edge family 안에서 최종 RF 시작 여유에 도달할 수 있다는 것이다. 경쟁 설명은 현재 family가 제한돼 실제 개선이 정체되거나 고정 여섯 회로 부족하다는 것이다.

같은 E281 RF case/원래 startup101/261 edges/783 coefficients, 세 source 성분과 decoder/신경 적분/해부학적 연결을 유지한다. E284 freeze/source-review/root result/peak-structure audit 및 inherited29 source/input/package/E280 qualification을 pin한다. 다른 다리·teacher·train·development·whole BANC/body/F6/UI는 사용하지 않는다. E284 재실행이나 다른 seed 선택은 없다.

최종 실제 판정은 E284 `strict_rf` 그대로다: q peak .0359964와 SSE1.091007285808255, LTM peak .0249975와 SSE.00801739628019, hard coefficient 및 triple interior, tolerance0, 최종 f32 실제 평가. RF 성공도 전체 초기화나 자연 보행 성공으로 해석하지 않는다. 이전 기각 기준/결과를 바꾸지 않는다.

## 변경 방법

최대6 outer, 각 시점의 fresh 실제 J로 변수 `x=[d,s]`인 sparse QP 하나를 푼다. 목적은 `.5*x@x + 100000*s`, s는 radians이며 `[0,.01]`이다. 중간 slack은 q peak 행에만 더해진다: `abs(qerr+Jq*d) <= .0359964-1e-6+s`. 최소 norm 목적이 불필요한 계수 이동을 억제하며 큰 고정 penalty가 peak 위반 감소를 우선한다. penalty/radius/slack cap sweep은 없다.

계수의 원래 hard domain `[0,8]`와 triple sum `[.125*(1+1e-6),8*(1-1e-6)]`는 유지한다. 각 step은 `d in [-.5,.5]`와 hard domain의 교집합이다. neutral 주변 box는 적용하지 않는다. 이 .5는 E277에서 사용하던 최대 local radius이며 E284보다 탐색 domain을 더 넓히는 실험이 아니다. q norm 및 LTM peak/norm은 E284 선형 한도와 1e-6 reserve를 그대로 사용한다. 따라서 중립에서 d=0과 충분한 s는 허용되고, 원래처럼 final peak infeasibility만으로 첫 이동을 차단하지 않는다.

P는 (n+1)차원 identity, q의 마지막 값만100000이다. A의 q peak 양쪽에 -1 slack 열을 추가하고 나머지 inherited constraints에는0열, 마지막에 s 상하한 두 nonnegative 행을 넣는다. SOC 순서는 nonnegative, q SOC, LTM SOC다. `b-A*x`, primal cone/residual>=-1e-8, dual cone>=0, stationarity infinity `|x+q+A.T*y|<=1e-8`, primal `.5*x@x+q@x`, dual `-.5*||q+A.T*y||²-b@y`, scaled gap[-1e-10,1e-8], complementarity<=1e-8*scale를 독립 검사한다. Solved/AlmostSolved만 허용한다. metadata/shape/nonfinite 오염을 기각한다. solver 출력 clip/repair를 하지 않는다.

각 QP 해 d에 alpha `[1,.5,.25,.125]`를 순서대로 시험한다. 실제 RF evaluation은 derivative=False이며 모든 실제 출력과 비용을 저장한다. raw coefficients가 hard domain/triple/step bound를 exact 통과한 경우만 평가한다. `predicted_peak=maximum(abs(qerr+alpha*Jq*d))`가 현재 actual peak보다 작아야 한다. 실제 수용은 q SSE/LTM/계수의 기존 strict 조건을 유지하면서 `actual_peak <= current_peak - 1e-4*(current_peak-predicted_peak)`를 만족해야 한다. 최종 peak 미달인 중간 점은 후보로 채택하지 않는다. 네 alpha 모두 기각되면 종료하며 반경·penalty 변경이나 extra retry는 없다.

수용된 실제점이 final strict를 통과하지 않으면 다음 outer 직전에 동일 계수를 새 engine으로 derivative=True 재평가한다. 직전 저장 q/LTM와 abs1e-10 재현, 같은 coefficient/error/target 관계를 확인한 후 그 J만 사용한다. 최종 strict f64 통과 때만 한 번 f32→f64 실제 평가한다. 최종 f32 strict 실패는 기각이다. 여섯 outer/actual 실패/uncertified QP/시간 초과 이후 추가 보정은 없다. 모델·보정·인증·저장·cleanup 실패의 부분 결과와 요청 비용을 보존한다.

## 판정과 다음 결정

최종 f32까지 통과하면 RF 초기화 방법만 지지되며 다음 여섯 다리 통합 설계의 입력으로 삼는다. 실제 감소는 있으나 최종 실패면 bounded 방법 기각/정체 위치를 기록한다. 첫 actual line search에서도 개선이 없거나 인증 불가능이면 이를 보존하고 decoder의 고정 전달 또는 source 시간 성분이라는 구조적 대안을 검토한다. 같은 하이퍼파라미터 재시도나 final 한도 완화로 성공을 만들지 않는다.

## 소유권과 자원

Actual gpt-6-sol/xhigh가 신규 `tools/banc_m3_rf_elastic.py`, `tools/verify_c_banc_m3_rf_elastic.py`, `tests/test_c_banc_rf_elastic.py`만 구현한다. Root FIRST→필요하면 동일Sol 첫수정→root SECOND→root 잔여 수정 순서를 따른다. 기존 source29개는 불변이다. 생성 CPU120초/tiny<=20/n<=6(neural coefficients), 원본/CUDA/큰 solver0. 작은 generated linear/nonlinear/failure case에서 중간 slack과 실제 감소·fresh J·alpha 순서·f32/엄격 gate·partial accounting·weakref 실제 engine 생존을 확인한다. 생성 비용/실패도 보존한다.

원본은 승인 후 한 번 science35초+cleanup5초/supervisor40초, RAM4GiB/GPU추가 추정2GiB/출력1GiB/free disk2GiB. 원본 누적7855.758807299546/7905초, 잔여49.24119270045412초 안에서 배정하며 현재예약0. 최대31 engines/평가(중립1+trial24+fresh5+f32 1), derivative6, forward62000/derivative12000ticks, 실제 engine1생존, QP6(각4초/100iter/thread1/Clarabel0.11.1/tol1e-8)이다. engine와 evaluator 참조를 해제한 후 cleanup/다음 constructor. Supervisor 전체 wall을 공유 누적에 한 번 반영한다. 원본 전 live 자원·source freeze를 확인한다. BANC/body/physics/fullfit/F6/후보 채택0이며 F1–F8은 계속 미완료다.
