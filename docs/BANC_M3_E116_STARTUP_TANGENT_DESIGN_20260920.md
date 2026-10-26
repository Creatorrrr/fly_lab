# E116: 초기 제약이 지속 구간 개선 방향을 막는지 저장 미분으로 진단

2026-09-20 · 부모 설계 · 구현과 첫 수정: Sol · 두 번째 리뷰 뒤 잔여 수정: 부모

E115는 원래 입력 fit 1회(외부101.516초/fit94.969초), 다리별500개 평가로 두 입력의 초기 제약을 만족했지만, 발 위치 오차 비율1.009438/1.005099와 발 속도 오차 비율.999657/.998062로 형태 개선에 실패했다. 부모의81개 독립 응답검사,3216개 저장자료/선택검사,81개 추가 지표/원본보존 검사가 이를 확인했다. E115를 채택하거나 실제 신체에서 실행하지 않는다. 마지막 실제 E113의 F3/F4/F5 실패와 E110의 F3만 통과한 부분 결과는 유지한다.

다음 질문은 **선택된 지점의 작은 국소 구간에서 초기99의8개 제약을 유지하는 것이 두 입력 중 나쁜 쪽의 형태 손실 감소를 얼마나 제한하는가**다. E115 fit을 더 오래 돌리거나 범위를 늘리지 않는다. 이미 저장한 미분만 사용하는 선형 진단이며 새 conditional response, CUDA, 신경/물리, calibration 생성 모두0이다. 선형 계획법(LP)12회는 별도 계산으로 명시한다.

## 고정 입력과 원점

e115-frozen-design, E115 부모 최종 검토·원본manifest·source review, E115 selected-delta, 여섯 selection JSON과 해당 best-feasible eval NPZ, parent-e115-data-review-raw를 해시로 고정한다. 선택된 index는 LF/LM/LH/RF/RH500, RM497다. 원자료 delta, parts(2×4), losses(2), ratios(2×4), jacobian(10×n)의 shape/finite/선택 일치를 검증한다. Jacobian의 첫2행은 loss, 나머지8행은 문맥 순서의 초기 qRMS/qmax/LTMRMS/LTMmax다.

이 원점은 E115의 **float64 solver 지점**이다. 최종 native float32 W의 재미분이나 실제 폐루프 상태로 부르지 않는다. 기존 E115 delta bounds와 다리별 partition을 유지한다. 분석용 증분 h는 각 변수에서 `[lower-delta, upper-delta]`와 `[-.01,.01]`의 교집합으로 제한한다. 빈 bounds나 원점이 범위 밖이면 계산 전에 실패한다. .01은 국소 선형 진단 범위이고 새로운 efficacy 채택 범위가 아니다.

## 다리별 두 선형 문제

변수 x=(h,t), t 하한0/상한없음. 원래 prior 상수 P=6e-5*(n/1572)*mean(delta²), 미분 p=2*(6e-5*n/1572)*delta/n이다. 두 문제 모두 선형 목적 `t + p·h`를 최소화하고 보고 때 P를 다시 더한다. 공통 두 제약은 `L_c + J_loss_c·h <= t`다.

1. `shape_box`: 위 두 제약과 h bounds만 사용한다. 초기 제약을 제거한 수학적 비교문제로서 이 해는 후보가 아니다.
2. `startup_box`: 위에 `ratio_j + J_start_j·h <= (1-1e-4)^2`의8개 제약을 각각 추가한다. startup 지표나 판정 cap은 바꾸지 않는다.

각 다리 두 문제를 정해진 순서로 한 번씩 풀고 총12회로 고정한다. SciPy1.18.1 `linprog(method='highs-ds')`, presolve=true, time_limit=1초, primal/dual feasibility tolerance=1e-9, threads=1을 사용한다. SciPy에서 threads를 HiGHS에 전달할 때 나오는 알려진 option-forwarding 경고는 보존하고 다른 경고와 구분한다. OMP/MKL/OPENBLAS=1. 인공 toy 문제만 소스검토 전에 실행하며 원래 gradient LP는 최종 리뷰 후1프로세스다.

각 LP 호출 전에 입력 c/A_ub/b_ub/lower/upper/행label/원자료hash를 저장하고 필수 started receipt를 쓴다. 저장 실패면 호출하지 않는다. 반환된 status/message/nit/fun/x 및 inequality/lower/upper residuals/marginals를 NPZ/JSON으로 **판정 전에** 보존한다. 원자료 저장 실패나 이후 오류는 가능한 수치 결과를 별도 NPY로 복구하고 원래 예외를 유지한다. started/returned/raw_saved를 구분한다. status1 시간/반복한도도 실패자료로 보존하고 재시도하지 않는다.

status0 결과에서 x/dual의 finite, bounds/inequality 잔차, dual 부호, stationarity, complementary slackness, primal-dual gap을 부모가 독립 검산한다. SciPy의 marginals 부호 규약을 사용하여 `c - A.T*m_ineq - m_lower - m_upper`가0인지 확인한다. unbounded t상한의 marginal은0이고 dual objective에 무한상한×0을 넣지 않는다. 수치 검증 허용오차는1e-8×(1+해당 비교량의 최대 절댓값)이며 이는 LP 수치검사에만 적용한다. 기존 생물학/보행/형태 gate는 불변이다. [SciPy 공식 문서](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html?highlight=shadow)의 시간 제한과 marginals 정의를 참고했다.

## 보고와 해석 한계

원점 worst loss+P, 두 LP의 최적 선형 목적+P와 감소량, startup 추가에 따른 최적 목적 차이, raw에서 계산한 2개 loss/8개 초기비율 예측, 각 bound 활성 개수, 제약 row별 slack/marginal을 다리별로 보고한다. shape_box의 초기 위반도 그대로 표시한다. 행마다 두 입력·네 초기지표를 명시하며8개를 평균내지 않는다. 선형 문제의 방향 h를 calibration으로 내보내거나 실행하지 않는다.

이것은 비선형 개선량의 보장이나 도달 가능한 보행의 상한 증명이 아니다. 특히 qmax의 미분은 현재 활성 sample의 국소 미분이므로 다른 sample의 최대값 전환을 예측하지 않는다. 제약 추가가 선형 개선을 줄이는지의 증거로만 사용한다. 이후 어떤 신경 시간응답 또는 모집 매개변수를 분리할지는 이 결과와 기존 실제 초기/지속 기록을 함께 보고 별도로 설계한다. 같은 weight-only fit의 반복 수/범위를 늘리는 승인은 없다.

구현은 새 tools/analyze_c_banc_m3_startup_tangent.py와 tests/test_c_banc_startup_tangent.py만 허용한다. 기존 모델/helper/원본을 수정하지 않는다. 다리별 미분·제약·dual 판정이 결합된 과학 계산 검토이므로 기존 명시적 gpt-5.6-sol/xhigh를 재사용한다. 작은 독립 구현은 새 agent 설정이 가능할 때 medium/high를 쓰되 기존 agent followup의 effort가 바뀌었다고 기록하지 않는다.

실제 LP 프로세스 내부20/외부30초, 총12 LP/각1초, 재시도0이다. 이전 E115 보수누적699.516초를700으로 올려 이번 단계의 별도 누적730초를 기록하며 이전 단계 한도838과 실패 판정을 소급 변경하지 않는다. 부모의 소스리뷰/저장자료 대수 검산은 별도 기록한다. 새 출력 e116-startup-tangent/만 사용한다. M2263/E110201/E113207/E11419/E1153043 원본, F6 미노출, 전체 F1–F8 active를 유지한다.
