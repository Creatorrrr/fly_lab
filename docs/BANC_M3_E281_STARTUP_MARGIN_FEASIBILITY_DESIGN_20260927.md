# E281 — 실제 전달을 위한 시작 오차 여유의 실행 가능성

2026-09-27 root 사전 설계. E280은 고정 여섯 제안 모두에서 비선형 보정의 효과를 확인했다. E277의 train 20%·LM 개발 기준 미달과 E279의 실제 시작 max .040869485 > .04 기각, 최종 F1–F8은 유지한다. 이 단계는 **다음 학습의 시작점이 실제 전달 여유를 가질 수 있는지**를 판별하며 보행 후보의 승격 단계가 아니다.

## 질문과 근거

E279의 조건부–실제 q 차이는 max .001516843/RMS .000136671이었다. 원래 학습의 경계 여유는 약 .00000390으로 부족했다. 다음 학습에 q peak .036, global RMS .0162를 내부 목표로 두면 기존 실제 기준 .04/.018까지 각각 .004/.0018 여유를 갖는다. 이는 관측된 후보 하나의 차이보다 크지만 새로운 후보의 실제 전달을 보장하지 않는다. 실제 기준을 완화하거나 native 확인을 생략하지 않는다.

중립 자체의 조건부 startup q max .0394578/RMS .0177271은 새 내부 목표를 넘는다. 그러나 저장 before Jacobian에서 |qerr|>.036인 모든 행은 계수에 대한 비영 sensitivity를 가진다. 이 사실은 국소 영향 가능성을 뜻할 뿐 제한된 계수로 동시 만족할 수 있음을 증명하지 않는다. 경쟁 설명은 기존 source 필터 자유도와 ±.05 영역으로는 충분한 시작 여유를 만들 수 없다는 것이다.

기존 중립을 이미 feasible이라고 가정하고 full fit을 시작하지 않는다. 먼저 동일한 51 source/1,660 기존 edge 필터 가족, 중립 초기 계수에서 학습 출력 없이 시작 구간만 최소 norm으로 보정한다. 이 단계가 통과할 때만 다음 full-fit 설계에서 그 초기화와 E280 보정을 결합할 근거가 된다.

## 고정 개입과 판정

LF/LM/LH/RF/RM/RH 순서로 각 다리 E273 exact CUDA startup101 context 한 개만 생성한다. E280 before의 full startup q/LTM를 abs 1e-10, q/LTM Jacobian을 abs 1e-9 이내로 재현한다. 이 자료와 명시적인 축 roster, 시작 target, physiology, source 자료를 SHA로 고정한다. 재현 실패는 invalid이며 해당 보정은 실행하지 않는다. 원본 6초 teacher·개발 데이터 평가·전체 BANC·신체 실행은 없다.

원래 99행 q SSE의 다리별 비율을 고정 가중치 `w_i=SSE_i/sum(SSE)`로 삼는다. q quota는 `w_i * 99*42*(.0162*(1-1e-4))^2`이다. LTM quota는 다리마다 `99*(.009*(1-1e-4))^2`로 동일하게 둔다. 이렇게 각 다리 quota를 모두 만족하면 합친 조건부 global 기준도 충족한다. q/LTM peak는 각각 `.036/.025*(1-1e-4)`다. q quota가 0이거나 자료/가중치가 비유한 경우 실행하지 않는다. 이 고정 할당의 실패는 모든 가능한 할당의 실패가 아니다.

각 다리 중립을 시작점으로 최대 두 번 fresh-J 최소 norm `.5*z.T*z` 보정을 한다. 계수 [0,8], triple interior [.125*(1+1e-6), 8*(1-1e-6)], 중립 기준 누적 trust box ±.05, 개별 correction maxabs .05를 유지한다. 현재 점이 새 startup 목표를 위반하는 것은 허용한다. 선형화 peak/norm 한도에서만 1e-6 절대 여유를 더 뺀다. QP/SOC와 독립 인증은 E280과 동일하다: Clarabel 0.11.1, 각 호출 2초/100iter/thread1/tol1e-8, Solved/AlmostSolved 후 exact primal/dual cone, stationarity, dual lower bound/gap, complementarity 검증. 인증 실패는 bounded rejection이며 세 번째 보정·line search·clip·자동 retry는 없다.

수정된 점을 실제 조건부 엔진으로 다시 평가한다. 원래 coefficient/interior/neutral trust box와 강화한 실제 q/LTM quota 및 peak를 **허용오차 없이** 만족할 때만 성공이다. 이 초기화 단계는 train 목적이나 Armijo를 사용하지 않으며 그 개선을 주장하지 않는다. 첫 보정 성공 시 조기 종료한다. 실패 시 현재 실제 점과 그 fresh Jacobian으로 두 번째 보정을 한다.

f64 성공한 다리에 한해서 계수를 한 번 float32로 반올림하고 다시 float64로 확장해 같은 CUDA 조건부 엔진에서 평가한다. 표현 변환 이외의 clipping/재조정은 없다. f32 계수도 원래 coefficient/trust와 강화한 실제 기준을 모두 만족해야 한다. f64 성공·f32 실패는 reject다. 실패한 다리의 추가 보정이나 다른 quota/범위를 시도하지 않는다. 유효한 scientific reject 뒤에는 나머지 사전 고정 다리도 진단하고, 계산·저장·재현 오류나 deadline은 전체 invalid/incomplete와 부분 원자료/비용을 보존하며 중단한다.

여섯 다리 모두 재현 및 f64/f32 강화 기준을 통과해야 `initialization_supported`다. 전체 실제 보행·train 20%·개별 development·F1–F8 통과가 아니다. 어떤 결과에서도 본 단계에서 full-fit 시작, 제품 후보 생성·채택, native/body/F6 실행은 없다. 일부 성공은 별도 기록하고 전체 승격은 하지 않는다.

## 구현과 검토

actual gpt-6-sol/xhigh가 네 신규 파일 `tools/banc_m3_startup_margin.py`, `tools/verify_c_banc_m3_startup_margin.py`와 대응 `tests/test_c_banc_startup_margin*.py`를 구현한다. Root FIRST → 같은 Sol 첫 수정 → root SECOND → root 잔여 수정 순서를 따른다. 기존 E280 및 inherited source, M2, UI는 수정하지 않는다. helper는 quota/problem/strict 판정을 명시적으로 받고 순수 배열+콜백으로 동작한다. E280 독립 인증/계측 재사용은 가능하나 원래 함수나 전역 CONTRACT를 monkeypatch하지 않는다. 강화된 peak/quota와 실제 저장 b의 일치를 검사한다. 각 단계의 current J, problem, raw solve/slack/dual, 실제 평가, f32 전환, 판정, 실패·완료 비용과 cleanup을 저장한다. 저장 실패가 성공이 될 수 없다.

Runner는 입력/소스/패키지/사전 quota/root 승인 SHA를 검증한 뒤 엔진을 생성한다. 명시 leg membership으로만 축을 줄인다. teacher 엔진 생성/평가, 전체 BANC나 body 생성, import 시 실행은 금지한다. 각 다리 엔진은 해제하고 GPU pool을 비운 뒤 다음 다리로 이동한다. deadline·evaluation/solver·memory/output 한도는 전체 단계에서 유지한다.

생성 CPU 검증은 q quota의 합과 global 의미, infeasible neutral의 정상 입력, 강화 경계·global/quota별 실패, 원래 coefficient/trust 유지, certified/uncertified solver, 두 보정 상한/조기 성공, f32 반올림으로 strict 경계가 깨질 때 거절, 재현 실패의 solver0, partial 비용·저장 예외·cleanup 보존을 포함한다. root는 원자료를 새 엔진 없이 독립 계산해 인증·실제 기준·f32 의미를 검토한다. Agent는 tiny 생성 CPU/mock만 실행하고 원본 solver/CUDA/BANC/body/fit은0이다.

## 자원과 다음 결정

현재 원본 누적7833.773759199601/7905초, 잔여71.22624080039895초를 유지한다. 한 번의 원본 supervisor55초(과학35+정리20), CPU1/CUDA1, RAM6GiB/GPU추가4GiB/output1GiB/free disk4GiB. 최대6조건부 엔진/max1live, 12SOCP, 24계수 평가, forward/derivative 각 최대48,000ticks. full fit/teacher/development/전체BANC/body/물리/F6는0이다. Source 승인 전 예약0이며 새 배정 증가0이다. 실패 비용과 생성·수동 검산 비용을 구분한다.

통과하면 다음 full-fit에서 강화한 시작 제약·고정 초기화·E280 보정을 적용하는 설계를 별도로 작성한다. 미달이면 이 제한된 초기화 가설을 기각하고 실패한 quota/peak 및 saved Jacobian을 확인한다. 같은 설계를 반복하거나 실제 기준을 낮추지 않는다. 어느 경우에도 기존 E277/E279 기각과 최종 자연 보행 미완료를 유지한다.

## 실행 전 E280 자원 근거 정정

E281 검토 중 E280 caller가 완료한 evaluate closure를 다음 builder 생성까지 보관하는 경로를 확인했다. 생성 weakref 검사에서 기존 두 객체가 남은 상태로 다음 두 객체가 생성되어 peak4를 보였다. 기존 meter의 peak2는 논리적 context 범위이며 실제 엔진 객체 생존 한도의 증거가 아니다. E280 실제 GPU 메모리 peak를 새로 측정한 것은 아니며 원본 수치·비용·6/6 독립 검산을 변경하거나 재실행하지 않는다. E280의 실제 생존 한도 준수 보장은 철회하고, 수치 근거만 다음 방법 설계에 사용한다. `parent-e280-resource-qualification.json`과 연결된 생성 증거를 E281 freeze에 명시적으로 포함한다.

E281은 단계 종료 시 evaluate 참조를 비우고 engine/evaluator 해제 뒤 pool 정리 및 다음 생성을 수행한다. 생성 weakref 회귀로 실제 참조 생존을 확인했다. Root FIRST는 이 수치 근거와 자원 한계 구분을 validator가 보존하도록 같은 Sol에 첫 수정을 요청한다. 이 정정은 원본 시작 전이며 강화 목표·입력·과학 가설·자원 배정은 불변이다.
