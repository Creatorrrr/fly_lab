# E206: 출발 제약을 명시하는 국소 신경 접선 적합

부모 사전 설계다. 목표는 같은 자연 보행 F1–F8이며 현재 미완료다. E203–E205의 조건부 반응식 적합은 각각 0.003727%, 0.184002%, 0.203507% 명령 개선으로 20% 후보 기준에 미달했다. E205의 마지막 점은 출발 최대 오차 .039999794rad로 .04 경계에 가깝고 비수렴 상태다. 단위 조정 뒤에도 높은 열 상관과 무반응 열이 남았다. 같은 CUDA 벌점 탐색의 무근거 연장을 피하고, 출발 경계를 직접 만족하는 접선 방향을 한 번 구한다. 경쟁 설명은 신경 응답의 국소 표현 범위가 작거나 선형 예측이 실제 비선형 반응에 전이되지 않는다는 것이다.

가설 H206은 저장된 E205 선택 접선에 명시적 제약을 적용한 한 번의 제한된 볼록 적합이 원래 대비 20% 개선을 예측하는 방향을 찾는다는 것이다. 이 가설은 국소 선형 계산에 한정되며 실제 신경 반응/보행을 주장하지 않는다. 더 풍부한 내부 상태를 가진 세포 모델은 구조 대안으로 유지한다. 현재 접선은 이미 있어 이 계산은 새 모델 구현보다 작고, 실제 개선을 막는 것이 벌점 방식인지 국소 가용 방향인지 구분하는 데 쓰인다.

## 고정 입력과 변경 범위

E201 실제 단계 입력/생리, E202 반응식, E150 양수 해부학 decoder, E197/E164 목표/행/초기99, 비LTM 343MN×7 계수, LTM 48MN w0, w[-1,1]은 유지한다. E205 선택 평가43의 weights와 네 신경 배열/decoder가 기준이다. E205 preconditioner의 343×7 power-of-two S를 그대로 사용한다. 전체 BANC/신체/CUDA/새 조건부 replay/새 교사/주기 입력은 0이다. F6 미노출, M2/E150/UI/실패·비용 보존이다.

새 코드 `tools/fit_c_banc_m3_constrained_tangent.py`와 `tests/test_c_banc_constrained_tangent.py`만 실제 gpt-6-sol/high가 소유한다. 제약 최적화와 원자료 회계를 연결하는 난도에 따라 high를 재사용하며 새 CUDA 동역학은 없다. 부모 FIRST → 필요시 같은 Sol 첫 수정 → 부모 SECOND/잔여 수정이다. E205의 검산된 `command_diagonal_scale`을 선택 상태에 호출하여 D만 얻는다. 이때 새로 반환되는 scale은 사용하지 않는다. E203의 순수 입력/metric 함수와 E202 loader/IO를 읽기 전용 재사용할 수 있다. 기존 gate/소스를 monkeypatch하지 않는다.

## 접선 모델

`delta_w = S*u`, `w_candidate = w_E205 + delta_w`다. D는 E205 선택 rate_sensitivity를 원래 decoder에 통과시킨 owned-muscle 접선이며 row0은0이다. 명령 접선은 `J[t,j,m,k] = .02*moment[j,membership[m]]*D[t,m,k]*S[m,k]`다. 같은 다리의 명령에만 기여함을 실제 moment/ownership으로 확인한다. 6개 다리별 자유 변수 인덱스를 보존하고 다른 다리 항을 임의로 버리지 않는다. 무반응 열도 원래 2401변수에 남기며 작은 정규화가 원점을 선호하도록 한다.

target 행 100..1923, velocity의 원래 인접 쌍, startup 1..99를 사용한다. 변수 u=0은 E205 선택 명령을 정확히 재현해야 한다. 실제 신경 초기 상태나 시간은 진행하지 않는다. 전체 예측은 `q_pred = q_E205 + .02*moment@sum_owned(D*delta_w)`이며 힘줄은 고정 LTM의 원래 값이다. 계산 중 새로운 신경 배열을 합성해 실제 response라고 저장하지 않는다.

목적함수는 `mean((q_pred[target]-target)^2)/(.8*original_command_RMS)^2 + 1e-8*sum(u^2)`이다. native 계수 bounds, 출발 표본별 최대 오차, 6개 다리 각각 command/velocity RMS 비악화, 전체 startup RMS를 명시적 제약으로 둔다. 다리별 velocity 비악화는 전체 비악화를 함의하지만 원래 전체 metric도 직접 재계산한다. 신뢰영역은 `norm(u)<=1`이다. 이는 원래 계수 가족의 전역 판정이 아닌 국소 방향의 범위다.

Clarabel 입력에서 RMS와 startup max, trust 한도에만 수치 여유 `1-1e-6`을 곱한다. 원래 판정은 완화하지 않으며 사후 원자료의 정확한 원래 한도로 다시 확인한다. 계수 bounds는 변환된 `(-1-w)/S <= u <= (1-w)/S`다. 목표20%는 목적함수 정규화/직접 판정으로 유지한다.

## 한 번의 원뿔 이차계획

Clarabel 0.11.1 직접 API를 쓴다. `P=2*Jq.T@Jq/(1824*42*(.8*baseline_RMS)^2)+2e-8*I`, `linear=2*Jq.T@Eq/denominator`이며 생략한 상수도 기록한다. 변수2401, 비음수 cone에는 2×2401 계수 bounds와 2×99×42 startup 표본 제약을 넣는다. 6 command SOC와 6 velocity SOC는 다리별로 작성한다. 각 norm의 raw J와 residual을 실제 RMS 한도로 정규화한 augmented Gram에서 대칭 eigendecomposition을 사용한다. 최소 고유값은 `-1e-10*max(1,max_eigenvalue)` 이상이어야 하며 음수 roundoff만0으로 바꾼다. 양의 고유값을 임의로 제거하지 않는다. factor.T@factor의 원래 Gram 상대 Frobenius 오차<=1e-10을 확인하고 값/부호/허용오차를 저장한다.

startup 전체 SOC는 원래 4158개 표본의 sparse J와 residual을 그대로 쓰며 별도 압축하지 않는다. trust SOC는 u 전체에 적용한다. 모두 합쳐 SOC14개다. `A*u+s=b` convention에서 norm cone의 s는 `[limit; offset+matrix*u]`이며 다른 부호 해석을 허용하지 않는다. P는 Clarabel용 upper triangular로 저장하면서 원래 대칭 이차식 검산이 가능해야 한다.

설정은 max_iter100, time_limit60s, max_threads1, direct_solve_method qdldl, verbose False, tol_gap_abs/rel/feas=1e-9다. 설치/업데이트/다중 초기화/선 탐색/추가 solve는 없다. callback은 iteration·잔차·현재 시간과 시간 한도를 기록한다. solver 원본 x/s/z/status/iteration/objective/잔차와 설정을 파생 계산 전에 저장한다. 반환 오류·시간 초과·비유한·출력 오류는 원래 오류 및 확보한 raw를 유지한다.

pre-solve에 D, S, baseline weights, variable/leg mapping, target와 startup residual, P/linear/A/b sparse 구성요소, cone 배열, norm factors/Gram 검산을 보존한다. 출력256MiB/free disk4GiB, resident memory 상한4GiB이며 최초 전체 J를 동시에 여러 벌 보관하지 않는다. 전체 target J는 다리별로만 구성한 뒤 해제하고 저장 D에서 재구성할 수 있게 한다.

## 검증과 판정

생성자료로 tangent 변수 순서/고정 초기행/해부학적 다리 경계, 압축한 SOC와 raw norm 일치, P/linear와 원래 residual 제곱합 일치, Clarabel 부호/출발 제약이 실제로 묶이는 알려진 작은 해를 확인한다. 생성 solve는 최대1회이며 원본 solve와 구분한다. 오류 주입은 원자료 보존/원래 오류/solve 1회 회계를 확인하는 데 한정한다. 부모는 원본 fit을 중복하지 않고 raw 접선 예측·제약·목적함수·출력 hash를 독립 재계산한다.

정상 종료 또는 유한 반복 상한에서 raw u와 w가 유한/범위 내이고 원래 출발·비악화·trust 제약을 만족하며 전체 command RMS가 원래*.8 이하라면 `supported_linearized_direction`이다. 후속 E207에서 실제 비선형 CUDA 반응을 별도 설계할 근거일 뿐 채택이 아니다. 20% 미달이면 `bounded_linearized_search_no_proposal`이고 전역 불가능성이나 자연 보행 반증이 아니다. 근사 해가 원래 제약을 넘으면 이를 그대로 별도 미달로 표시한다. 유효한 raw가 없는 오류/시간·메모리·출력 실패는 invalid/inconclusive이며 수치만 바꿔 자동 재시도하지 않는다. 충분한 개선 방향이더라도 원래 실패 기준은 유지하고 필요시 별도 실패 후 진단 계약으로만 비선형 검증을 설계한다.

이번 원본 process1/CPU1/GPU0/Clarabel solve1/retry0, 전처리+solve+저장 내부100/외부110/감독120초다. 이전 누적5401.354468499707/5444초를 유지하며 별도 명시적 제약 시험에120초를 추가 배정해 누적 상한5564초로 정한다. 사용자 지정 전체 예산은 없으며 과거 사용량을 초기화하지 않는다. 준비/생성/부모 검산은 별도 기록하고 미측정을0으로 부르지 않는다. 원본 결과와 남은 자원을 기록한 뒤 다음 접근을 판단한다. 최선E150/F1–F8 미완료/goal active는 유지한다.
