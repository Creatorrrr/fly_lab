# E284 — RF의 원래 계수 영역에서 시작 출력 capacity 진단

2026-09-27 root 사전 설계. E281/E282/E283 초기화 기각, E277 train/LM 기각, E279 native 시작 기각과 M3 finite-goal F1–F8을 유지한다. 이 단계는 실제 보행이나 전체 초기화의 수용 판정이 아니다.

E283은 중간 목표에서 실제 RM/RH peak 초과로 두 보정 한도 내 기각됐다. 별도 저장 dual 검산에서 중간 보정 후의 RF fresh J도 neutral±.05 안에서 최종 peak .0359964에 도달하지 못하며, .1로 넓힌 계산도 같은 witness가 기각했다. 중간 RM/RH에 세 번째 보정만 추가하는 방법은 이 RF 제한을 해결하지 않는다.

H284는 고정 세 source 성분의 RF 출력이 원래 허용된 계수 hard domain 안에서 최종 내부 시작 한도에 도달하지만, 중립 주변의 작은 box가 이를 막았다는 것이다. 경쟁 설명은 더 넓은 영역에서도 선형 충돌이 남거나 실제 비선형 응답 때문에 네 번 안에 도달하지 못한다는 것이다. 후자는 이 bounded 방법의 기각이며 전체 비선형 계수 family의 불가능성 증명이 아니다.

## 변경과 불변

E281 RF case와 E283 이전부터 사용하던 RF startup101/context/기존 edge261/계수783/targets/q7/LTM1만 사용한다. 다른 다리의 최적화나 실제 평가를 추가하지 않는다. 원래 neural engine/decoder/세 필터 성분/해부학적 연결은 그대로다. 기존 frozen source26개는 수정하지 않는다.

이번 진단에서만 **원래 hard coefficient domain 전체**를 탐색한다: 각 값 `[0,8]`, 각 triple sum `[.125*(1+1e-6),8*(1-1e-6)]`. E281–E283에서 도입한 neutral±.05와 step±.05 box는 이 실험에 적용하지 않는다. 결과는 해당 이전 box를 통과한 것으로 기록하지 않는다. 최소 norm correction 목적 `.5*d@d`와 hard domain이 보정을 제한하며 별도의 반경 sweep·clip·line search·재시도는 없다. 이 변경은 탐색 방법에 관한 것이고 actual startup 한도/F1–F8을 바꾸지 않는다.

실제 RF q peak `.036*(1-1e-4)`와 q SSE `4158*(.0162*(1-1e-4))**2`, LTM peak `.025*(1-1e-4)`와 SSE `99*(.009*(1-1e-4))**2`를 유지한다. q SSE는 전체 q budget과 같은 값이나 RF만 관측하므로 **전체 q norm 통과를 주장할 수 없다**. 이 진단의 주요 판별 대상은 RF peak이다. f32 표현 가능 `[0,8]` endpoint와 원시 해의 exact hard domain/triple 검사 후에만 실제 평가한다.

QP는 RF J만 사용하는 sparse minimum norm 문제다. q/LTM peak 및 SOC에는 기존 1e-6 선형 여유를 사용한다. `P=I,q=0,A*d+s=b`; nonnegative와 두 SOC. 새 계수 `c+d`에 대한 box/triple, qerr+Jq*d, lerr+Jl*d를 명시해 구성한다. 저장 P/A/b와 RHS/row order/sizes를 원자료에서 독립 재구성한다. Frozen E282 수식을 production monkeypatch하거나 동작하지 않는 trust metadata를 재사용하지 않는다.

인증은 원래 `b-A*d`의 모든 제약과 primal cone>=-1e-8, dual cone>=0, stationarity infinity<=1e-8, primal `.5*d@d`, dual `-.5*||A.T*y||²-b@y`, gap `[-1e-10,1e-8]*(1+max(abs(primal),abs(dual)))`, 상보성 absolute<=1e-8*같은 scale를 요구한다. `Solved/AlmostSolved`만 허용하며 nonfinite/형태 불일치/metadata 변경을 기각한다. 실제 hard coefficient와 출력 판정은 tolerance0이다.

## 고정 순서와 판정

1. RF 중립의 전체 q/LTM와 두 J를 E281 저장값에 abs1e-10/1e-9로 재현하고 coefficient/error/target 관계를 exact 확인한다.
2. 현 실제값이 최종 strict 기준을 충족하지 않으면 fresh 실제 J로 QP를 만든다. 최대로 네 번만 보정한다. uncertified QP나 strict coefficient 위반은 곧바로 중단한다.
3. 각 raw correction을 수정 없이 적용하고 새 RF engine 하나로 실제 q/LTM/J를 평가한다. 실패 점도 원자료로 저장하고 다음 fresh J 보정에 사용한다. 마지막 실제 strict 통과가 필요하다.
4. f64 통과한 경우에만 한 번 f32→f64 계수를 실제 derivative=False로 평가한다. 이 단계까지 통과하면 `rf_capacity_supported`; 전체 초기화/제품 후보 채택/자연 보행 성공을 뜻하지 않는다. 실패와 실행/저장 오류는 각각 rejected와 invalid/incomplete로 구분한다.

지원되면 이 원래 계수 영역을 전체 startup 초기화와 제한된 full-fit 방법에 통합하는 다음 설계를 검토한다. 기각되면 동일 radius 반복 대신 RF 입력 시간 성분 또는 startup 근사와 실제 신체 전달의 구조를 재검토한다. 어느 경우에도 E283 실패를 지우거나 E284가 F3–F8을 증명했다고 표현하지 않는다.

## 소유권과 자원

Actual **gpt-6-sol/xhigh**가 신규 helper `tools/banc_m3_rf_capacity.py`, runner `tools/verify_c_banc_m3_rf_capacity.py`, test `tests/test_c_banc_rf_capacity.py` 세 파일만 작성한다. Root 설계→Sol 구현→root FIRST→필요하면 같은 Sol 첫 수정→root SECOND→root 잔여수정 순서다. 생성 CPU검사120초/tiny<=20회/n<=6, 모델·원본·CUDA·큰 solver0에서 FIRST를 제출한다. 실제 요청/완료/반복수/시간과 오류를 보존한다. Root 생성 검사는 별도 비용이다.

Runner는 E282의 검증된 per-evaluation fresh engine 방식과 E281 StartupEvaluator/meter를 재사용한다. Wrapper는 RF input/case 하나만 넘기며 production RF axes가 원래 roster와 정확히 일치하는지 확인한다. 실제 live engine1, engine와 evaluator 참조를 해제한 뒤 cleanup과 다음 생성이다. max engines/evaluations6, derivative evaluations5, forward12000/derivative10000ticks, QP4(max4초/100iter/1thread/Clarabel0.11.1/tol1e-8). 각 단계/constructor/solve/certificate/cleanup/partial 비용을 모두 보존한다.

E283 freeze/review/root audit/RF saved-J bound/radius sensitivity, 기존26 source, input/case/roster/package SHA 및 E280 lifetime qualification을 고정한다. 승인 전 원본예약0. 승인 후 단회 science20초+cleanup10초/supervisor30초, RAM4GiB/GPU추가 추정2GiB/output1GiB/free disk2GiB이다. 원본 누적7853.549514399554/7905초와 잔여51.450485600446초를 공유하며 배정 증액0이다. 시작 직전 실제 자원과 다른 프로젝트 작업의 GPU 사용을 재확인한다. whole BANC/body/physics/teacher/train/development/fullfit/F6/UI/제품 후보 생성·채택은0이다.
