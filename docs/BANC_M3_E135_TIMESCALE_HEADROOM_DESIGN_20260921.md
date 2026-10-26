# E135: 초기99 제약 아래 시간상수의 국소 개선 여지

2026-09-21 KST · 부모 설계 · 원본 실행 전 계약

E134는 전체391 운동신경/96근육의 시간상수 감도를 독립 검증했다. ell=0의 LF rate/activation/B 초기99가 E132 선택39와 정확히 일치한다. E133의 기존 gain/edge 국소 예측은5% box 반경에서0.0330625% 개선에 그쳤다. H135는 같은 점에17개 LF logtau 자유도를 추가할 때 동일 반경에서 원래 최악 문맥 목적함수의 선형 예측 감소가1% 이상인지 확인한다. 이는 실제 비선형 개선이나 보행 성공의 대용물이 아니다.

## 그대로인 점과 추가 열

E132 선택39의 delta273/gain17, 손실/초기99 q·LTM 오차/원래 bounds, max-context epigraph 목적과 원래 delta/gain prior를 그대로 쓴다. ell17의 시작점은0, bound는 정확히[-math.log(3),math.log(10)]이다. E134와 동일 T=.3*exp(ell)의[.1,3]초 공학적 범위다. 새로운 prior를 이번 선형 진단에 추가하지 않는다. 전체 좌표는 edge273+gain17+ell17+epigraph1=308, 부등식4761이다. optimizer cap .99960004, selector .9998000100000001, 원래 shape20%/F1–F8은 유지한다. 출발점의8개 엄격 optimizer 위반도 숨기지 않고 같은 LP가 복구하도록 한다.

E134 zero_tau 원본의 global B열1668+LF근육index를 골라 tau_B[3,2001,17]을 만든다. E132의 저장된 activation cotangent[3,2001,17]과 곱해 time축을 합한 것이 tau_loss_jac[3,17]이다. q_tau_jac=tau_B[:,1:100,None,:]*기존 q_coefficient, LTM_tau_jac=tau_B[:,1:100,:]*ltm_coefficient*해당 local LTM근육 mask다. 이17열을 기존 Jacobian 뒤, epigraph 앞에 추가한다. 현행 E128 constraint_values를 재사용하고 기존 source는 수정하지 않는다. gradient는 기존290 prior 미분, tau0의17개0, epigraph1 순서다.

같은 점임을 먼저 검증한다. 기존 E133 reconstruct 결과 및 봉인 linearized-base의 vector/bounds/c/gradient/ratios를 대조한다. 새 Jacobian의 기존290열 및 마지막 epigraph열은 E133과 절대차1e-12 이하여야 한다. E134 zero의 LF rate/z/gain감도/activation/q 앞7행/LTM LF값은 E132 선택39와1e-8 이내, branch exact여야 한다. 기존 B 초기99는1e-8 이내여야 한다. 또한 E134의 기존 edge/gain 전체시간 B와 E132 cotangent로 재구성한 기존 loss_jac290열이 원래 값과1e-10 이내인지 확인한다. 새 tau열은 실제 B와 activation_log_tau_sensitivity의 일치, ID·membership·shape·dtype·finite를 확인한다. 이 점검 실패를 solver 성공으로 덮지 않는다.

## 고정 실행과 판정

부모 최종 소스 리뷰 뒤 CPU HiGHS dual simplex, threads1/presolve, primal·dual1e-9/time_limit1초로 반경 .005,.02,.05의3개 LP를 한 번씩 실행한다. 기존 box와 |step_j|<=rho*(upper_j-lower_j)를 교차하며 epigraph step은 무제한이다. 선형 부등식은 -J*step<=c이고 min gradient·step이다. primary는 사전 고정 rho=.05, predicted decrease fraction=-objective_change/base_true_worst_objective >=.01이다. 선형·bound 잔차<=1e-7, objective 일치<=1e-9와 optimal status가 필요하다. 나머지 반경 결과로 primary 실패를 대체하지 않는다. E133 저장 primary 값을 비교 보고하지만 옛 LP를 재실행하지 않는다.

재구성된 전체 matrix/vector/gradient/tau_B/손실 및 startup tau Jacobian, 각 LP의 답/slack/bounds/dual·marginal/상태/시간을 수치 판정 전에 저장한다. 요청 직전/반환 직후 flags, 실제 요청·반환·미확정·반환후 저장오류 카운트를 모든 종료 경로에 남긴다. 원본은1프로세스/LP3개/재시도0이며 기술 오류 시 다음 LP를 시작하지 않는다. partial answer와 primary/secondary 오류를 보존하고, 마지막 기록 실패 후 성공을 보고하지 않는다. 단순 generated 시험이나 부모 수학 검토에 원본 LP를 숨기지 않는다.

새 신경·CUDA·geometry·신체·GraphStore·StateStore·적합 실행은0이고 이미 노출된 E110/E113/E121 저장 데이터만 읽는다. M2/활성 UI/F6 두 자세를 보존한다. 원본 내부8/외부10/supervisor12초, 다음 LP 이전2초를 보존한다. E134까지 보수적1573.5490344999707초를 보존하고 새 사전 기준1574+12=1586초로 배정한다. 누적1742초 상한 안의 별도 H135 계약이며 종료된 E134의15batch 또는 과거1530초 계약을 연장/수정하지 않는다. 준비·합성·리뷰 비용도 별도 보존한다.

부모는 설계·동결·독립 Jacobian/LP 잔차 재검증·원본 supervision을 소유한다. 실제 gpt-5.6-sol/high가 새 tools/analyze_c_banc_m3_timescale_headroom.py와 tests/test_c_banc_timescale_headroom.py만 구현한다. 새 동역학 미분은 E134에서 검증됐고 이번은 저장 감도/제약/기록 통합이므로 high를 쓴다. 부모 첫 리뷰→필요하면 같은 Sol 첫 수정→부모 재리뷰→잔여 부모 수정 순서다. Sol은 작은 generated fixture만 시험하고 원본을 실행하거나 재위임하지 않는다. 기존 E133/E134/E128 source/freeze/원자료는 불변이다.

H135가 supported이면 같은 초기99/목적함수를 갖는 실제 nonlinear LF 제약 적합을 다음 별도 계약으로 검증한다. 그 뒤 전 다리·native158706·동일 후보 신체/F1–F8이 필요하다. refuted/inconclusive이면 근거에 맞는 다음 동역학/적합 설계를 택하며 M3를 완료 처리하지 않는다. LP step 자체는 보행 후보로 채택하거나 UI에 적용하지 않는다.
