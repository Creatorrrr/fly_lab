# E212: 시간 제약을 푼 해부학적 전달 용량 진단

E211의 고정 반응 기저는 전신 startup 조건에서도 PrimalInfeasible였다. 별도 저장 자료 진단에서 관절 하나씩의 낙관적 범위에는 startup4158개 목표가 모두 포함됐고 main 거리 RMS 하한은.000107758이다. 이것은 관절 간 공통 근육과 시간적 일관성을 무시하므로 무엇이 실패 원인인지 아직 구분하지 못한다. 다음 한정 진단은 근육·관절의 동시 소유권과 양의 출력 상한은 유지하고, 시간에 따른 같은 신경 계수 사용 조건만 풀어 전달 용량을 측정한다. 새 신경 후보나 외부 제어기를 채택하는 작업이 아니다.

## 고정 원자료와 하나의 이완 문제

E209 `affine-readout.npz`의 D(2001,391,16), q_intercept와 E150 start의 membership391/moment42×96, E203 free343/fixed48, E197 동일 목표를 고정한다. D>=0/fixed48 D=0과 다리 간 moment0을 확인한다. 각 자유 근육 g의 추가 activation 상한은 U[t,g]=sum_{m:membership[m]=g,k}D[t,m,k]다. 고정48 LTM의 효과와 canonical 초기 activation/동차응답은 q_intercept에 남는다. 실제 rate나 cap을 고치지 않는다.

frame row1..1923은 startup99개와 main1824개 목표를 정확히 이어붙인 것이다. 6다리×1923frame의 독립 문제에서 `C=.02*moment[leg7,free_muscles_of_leg]*U[row,group]`, `y=target[row,leg7]-q_intercept[row,leg7]`로 정의한다. 독립 fraction v∈[0,1]을 사용해 `min ||C v-y||²`를 푼다. 추가 activation은 U*v다. fraction은 단위 정규화이며 물리 상한을 바꾸지 않는다. 상한이 정확히0인 근육은 변수에서 제외하되 약한 양수 변수는 제거하지 않는다. 모든 상한이0이면 solver 없이 v0의 값을 직접 기록한다.

이는 같은 근육이 해당7관절에 동시에 작용하는 관계를 보존하지만, 각 frame의 activation을 별개로 고르는 낙관적 이완이다. 신경 기저의 시점 간 동일 beta, rate/activation 도함수 제약, actual sensory feedback를 제거했다. 따라서 낮은 오차나 가능한 posture가 나와도 신경계 학습·실제 보행 성공이 아니다. 현재 gate/tau/cap에서 기록된 입력의 endpoint envelope 안이라는 범위만 유지한다.

## 사전 고정된 solver와 증명값

SciPy `lsq_linear(method='bvls', bounds=(0,1), tol=1e-10, max_iter=100)`를 원본 각 비어 있지 않은 문제에 최대1회 적용한다. leg LF/LM/LH/RF/RM/RH, frame 오름차순으로 실행하며 최대11538호출이다. 원본 프로세스1, warmup/튜닝/원본재시도0이다. 기존 E211/Clarabel의 재시도가 아니며 서로 다른 사전 정의된 이완 문제다. 반복 cap 또는 오류가 생기면 raw/완료 frame을 보존하고 전체 진단은 incomplete로 구분한다. 누락된 해를0으로 보충해 전체 성과를 계산하지 않는다.

반환값은 실제 box 내에 있는지 먼저 확인하고 clip하지 않는다. 각 문제의 r=Cv-y, f=r^Tr, gradient=2C^Tr와 `L=f-gradient^Tv+sum(min(gradient_i,0))`를 저장한다. 목적의 convexity에서 이 L은 box 전체 최소값의 affine minorant다. 알려진 f>=0도 사용해 max(0,L)를 lower bound로 기록할 수 있다. 사후 성과를 높이는 보정이 아니다. 반환된 feasible v의 f는 upper bound이고 두 값의 gap은 수치 최적성 진단이다. lower>upper+1e-10 또는 gap>1e-8*max(1,f)이면 certified로 표시하지 않는다. 실제 scipy status/optimality/nit와 호출·시간을 함께 남긴다.

원시 U, group/motor 소유권, frame/목표/q0, v·activation·q, residual, f/L/gap/status/iterations/solver 호출 수를 저장한다. 전역 dense Jacobian을 만들지 않는다. 전체 main과 각 다리에서 sqrt(sum L / 해당 관절 표본수) 및 feasible returned error RMS를 함께 보고한다. startup RMS·max와 동시 기준 통과 여부를 별도로 보고하며 최소 L2 해의 개별 최대오차가 큰 것만으로 다른 해도 불가능하다고 주장하지 않는다. 반환 posture들의 main velocity도 기술 통계로 계산하되 실제 causal motion이라고 부르지 않는다. 표본·목표 재정렬/새 위상/창 변경은 없다.

예상 가능한 판정은 세 가지다. (1) 수치 certified 하한이 원래 q/startup RMS proxy를 넘으면 이 이완에서도 부족하다는 근거다. (2) 상한 내 posture가 원래 q/startup 조건을 만족하면 동시 해부학 전달의 존재 증거이고 시간적 신경 제약/동역학이 다음 연구 대상이다. (3) lower와 feasible upper가 기준을 가르면 unresolved다. 어느 경우도 F1–F8 완료나 생물학 검증이 아니다.

## 구현·검토·실행

부모 설계→실제 gpt-6-sol/high 구현→부모 FIRST→필요시 동일Sol 첫수정→부모 SECOND/잔여수정 순서다. convex lower certificate·표본 정렬·11538회 한정 호출·부분 raw 보존이 있어 high를 사용한다. 소유 파일은 `tools/analyze_c_banc_m3_anatomical_capacity.py`, `tests/test_c_banc_anatomical_capacity.py`뿐이다. 기존 E211/E209 소스와 결과를 수정하지 않는다.

cheap 생성/모의 검증은 그룹 cap 합, 소유권, box minorant, 원시 저장 실패/중단을 확인한다. 부모 리뷰 후 실제 생성 BVLS는1회만: C=I2,y=(1.5,-.5), box[0,1]의 분석해 v=(1,0), f=lower=.5, gap0을 검증하고 전체 반환을 판정 전에 저장한다. 원본 예비 풀이/원본 테스트는 없다.

원본 CPU1/GPU0/process1, 최대11538 BVLS calls, 내부60/외부80/감독90벽초, RSS2GiB/출력64MiB/free4GiB다. 1호출 전에 시간/카운터 한도를 검사하고 끝난 호출의 raw를 즉시 메모리에 보관하며 주기적으로 완료 구간을 디스크에 보존한다. 외부 종료에서 저장되지 않은 마지막 구간의 비용/결과를 추측하지 않는다. 독립 parent 검산은 기존 저장 결과만 사용하고 새 BVLS를 호출하지 않는다.

원본 누적5636.023455099708/배정6004초, 잔여367.976544900292초에서90초 예약한다. 추가 원본 배정0. 준비/생성/부모 검산은 별도 계상하고 미측정을0으로 세지 않는다. 새 CUDA·전체BANC·신체0, F6미노출/최선E150/M2/UI·실패기록 보존. F1–F8 active·미완료를 유지한다.
