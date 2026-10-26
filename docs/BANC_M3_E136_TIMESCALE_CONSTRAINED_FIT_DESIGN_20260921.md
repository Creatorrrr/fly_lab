# E136: 초기 자세를 보존하는 실제 시간상수 제약 적합

2026-09-21 KST · 부모 설계 · 원본 실행 전 계약 · M3 pending

이전 E135는 유효한 진전이다. 고정 반경에서 1% 개선을 예측한다는 H135는 0.725876%로 미달했으며 이 판정을 유지한다. 저장된 답을 추가 해석하니 logtau17개 모두 반경 경계에 있고, 반경에 대한 목적함수 개선 감도의 대부분이 이17변수에서 나왔다. E135에서 실제 탐색한 T는0.253–0.356초로, E134에서 이미 검증한 공학적 허용 범위0.1–3초보다 좁다. 동시에 일부 초기 관절 제약은 edge/gain/tau 변화가 서로 상쇄하며 유지된다. 따라서 초기 제약을 풀거나 LP 반경을 바꿔 H135를 다시 판정하지 않는다. 다음 별도 H136에서는 원래 허용 범위에서 제약을 매번 다시 계산하는 실제 비선형 적합을 검증한다.

## 가설과 범위

H136은 E132 선택39의 LF feasible점에 ell17=0을 추가한 시작점에서, E134의 실제 CUDA 적응 동역학을 사용한 bounded SLSQP가 같은 세 기록 문맥의 원래 worst objective+prior를1% 이상 줄이고 초기99 selector 제약을 보존한다는 것이다. 가장 강한 대안 설명은 LP 감도가 작은 반경에서만 유효하거나, gain/edge/tau 사이의 초기 자세 상쇄가 비선형 계산에서 깨져 유효한 개선이 사라진다는 것이다.

Primary는 사전에 고정한 단일 적합에서 기록된 selector-feasible 결과 중 원래 true_worst_objective 최소값이다. 같은 값이면 가장 이른 원본 index를 선택한다. 첫 실제 replay를 기준으로 objective ratio<=.99이고 선택index가1이 아니면 H136 supported이며 LF 조건부 proposal만 채택한다. valid한 bounded 적합에서 미달하면 refuted다. 원본/기하/해시/초기 동일점/오류/자원 기록이 불완전하거나 기술 오류가 나면 inconclusive/invalid로 분리한다. SLSQP 종료상태가 bounded-stop이어도 모두 저장된 유효한 평가에서 선택할 수 있고, 실패하거나 저장이 누락된 평가를 제외하고 성공을 선언하지 않는다.

목적함수·teacher/mask/geometry·기존 delta273/gain17 bounds·optimizer cap .99960004·selector cap .9998000100000001·초기99·기존 whole shape20%/F1–F8은 유지한다. 새 prior는 없다. prior는6e-5*sum(delta²)/1572+1e-5*sum(gain²)/96이며 ell 미분17열은0이다. T=.3*exp(ell), 정확한 ell bounds는[-math.log(3),math.log(10)]으로 E134와 같다. 변수 순서는delta273/gain17/ell17/epigraph1이다. 외부 신경 파형·새 연결·root 교정·제품 UI를 추가하지 않는다.

## 구현

새 `tools/fit_c_banc_m3_timescale_constrained.py`와 `tests/test_c_banc_timescale_constrained.py`만 구현한다. E134의 `CudaAdaptationTimescaleEngine`을 LF69MN/17근육/273edge로 생성하고, E128의 고정 fixture/geometry/기하 손실과4761 constraint 함수를 재사용한다. 기존 source 전역 monkeypatch는 금지한다. E128의 gain-only `evaluate_leg`, `activation_cotangent_product`, `objective_and_jac`, `solve_one_leg`를307변수로 그대로 호출해서는 안 된다. 새 evaluator/감도곱/목적미분/solver closure가 ell 축을 명시적으로 소유한다. bounds, cache의 epigraph 제외 byte key, 원래 `FitLimit` 예약시간은 유지한다.

손실은 기존 q·foot·foot velocity·LTM 네 성분과 정확히 같은 fit_time_mask를 사용한다. CUDA activation을 DLPack으로 Torch leaf에 공유하고 geometry loss의 cotangent를 구한다. E134 parameter-to-local-muscle map으로 full B와 곱해307열 loss gradient를 CUDA에서 축약한다. 초기99 q/LTM 감도는 같은 B와 기존 moment/tanh 미분에서 구한다. 실제 CUDA 장치와 CPU/BLAS/Torch thread1, fixed/dynamic upload, 공유/복사, geometry forward/backward, 신경 context/edge-bank work를 기록한다.

각 원본 평가마다 raw에 delta/gain/logtau·신원·11응답/branch·full B·B_initial99·cotangent·losses/parts/loss_jac·초기오차/계수/ratios·각 문맥의 fit feet를 저장한다. 추가 full B 보존은307열 chain rule을 부모가 모든 평가에 대해 새 신경 호출 없이 검산하기 위한 것이며 원본 raw 최대4GiB를 배정한다. raw/partial을 수치 판정 전에 저장하고 파일 해시를 기록한다. 최종 선택점의신경 재실행은 하지 않는다.

처음의 실제 replay는 E132 선택39의 rate/z/gain감도/activation/command LF7행/LTM/branch와 일치해야 한다. 출력/B 기존290열의 허용차1e-8, losses/parts/ratios와기존loss_jac는1e-10, branch/ID는exact다. 추가tau열의초기B와loss_jac는 E135 저장tau열과1e-10 이내로 비교한다. E135와 같다는검사에실패하면 optimizer를시작하지 않는다. baseline재실행도최대64평가중1개로계수한다.

## 단일 실행·중단·비용

원본은1프로세스·SLSQP1회·최대64unique평가·재시도0이다. solver옵션은maxiter100/ftol1e-9/dispFalse로 고정한다. warm start는 E13239+ell0 하나이며 E135 LP step을 다른 초기점으로 사용하지 않는다. 매 평가 시 현재 비선형 constraint를 다시 계산한다. 신경 계산 반환과raw저장 완료를구분하고, 요청 직전의durable intent/반환 직후의known progress/최종raw완료를각기 남긴다. 반환 후IO오류가 이미 알려진 신경 work를unknown으로바꾸어서는안된다. cache/solver/시작전오류, primary/secondary오류, 미실행·미확정·반환후오류를모든종료경로에서보존한다. 기술오류후다음원본평가를하지않고, 마지막ledger/manifest실패후성공을보고하지않는다.

실제자료와무관한1MN/1근육/1edge/1문맥20context tick의tinyCUDA1회(.005조건부초,1derivative bank)를 setup에서별도기록한다. E134계산기는이미독립검증되었으므로새FD sweep을추가하지않는다. 실험에는새full391전체응답/full158706 BANC/신체/F6/UI가없다.

fit95초,다음평가이전leg reserve3초/process reserve5초,전체내부108초/외부116초/supervisor120초다. 최대64평가는1920조건부초/7,680,000context tick/529,920,000scalarMN tick/8,386,560,000edge-stage tick/64bank/384,000decoder endpoint/350,272geometry forward 및같은backward row다. E135까지1577.387804900005초를보존하고새사전기준1578+120=1698<=1742초로배정한다. 종료된E134/E135및과거1530초계약을연장하거나실패기록을초기화하지않는다. 준비/합성/리뷰비용은별도보존한다.

부모가설계·입력동결·독립baseline/gradient/selection검토·원본supervision을맡는다. 실제gpt-5.6-sol/high가첫구현과필요한첫수정을맡는다. 새CUDA동역학은E134에서검증되어있고이번은추가307열/evaluator/제약최적화의통합이므로high를선택한다. 부모첫리뷰→같은Sol첫수정→부모두번째리뷰→잔여부모수정순서를따른다. Sol은작은generated CPUfixture로만시험하고원본을실행하거나재위임하지않는다.

## 다음 결정

LF proposal성공은M3나생물학적보행의완료가아니다. 다른다리의초기/형태적합과동일후보의nativeBANC·신체F1–F8검증이남는다. 미달하면64평가를이어붙이거나한도를늘려같은실험을재개하지않고, 실제곡률·상쇄·활성제약의기록에근거해다음구조/목표전달설계를판단한다. 기존M2 263파일,활성UI,미사용F6두자세를보존한다.
