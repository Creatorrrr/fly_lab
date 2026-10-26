# E204: 같은 국소 반응식의 변수 단위 조정

부모 사전 설계다. E203은 유효한 제한 탐색이었지만 첫 선 탐색 maxls10에서 iteration0으로 끝났고 명령 개선은 .003727%였다. 계수 최대1인 첫 단계의 merit는 2.23e13인 반면, 약3e-10/1e-9의 작은 단계는 해석적 기울기의 감소 예측과 일치했다. 따라서 모델 전체를 기각하지 않고 탐색 단위가 부적절하다는 설명을 시험한다. 경쟁 설명은 조정 후에도 보호조건과 명령 형상이 양립하기 어렵거나 국소 탐색이 충분하지 않다는 것이다.

## 고정된 비교와 개입

E203의 입력97개, 실제 E201 단계 전류/생리/초기 상태, E202 계산기, E150 해부학적 decoder, 고정 LTM48/가변343 MN, 7기저, 보정폭.5, w범위[-1,1], 2401변수, 목표/시계/초기99/판정 기준은 그대로다. E203 도구와 시험은 수정하지 않는다. 원래 신경망·신체 실행0, F6 미노출이다.

원래 E203 evaluation-000의 실제 gradient_w를 읽어 `S = 1/max(abs(gradient_w))`를 동결한다. S=0.0003115579064523788이다. 새 변수 u에서 w=S*u로 계산하고 u범위는 정확히[-1/S,1/S]로 변환한다. 정규화나 사후 clipping으로 w범위를 줄이지 않는다. 목적함수는 원래 w에 적용하며 optimizer에는 gradient_u=S*gradient_w를 넘긴다. E203 평가/선택/실패는 보존하며 초기점은 다시 w0 한 개다. seed/restart는 없다.

SciPy1.18.1 L-BFGS-B1회, jac=True, maxiter40/maxfun60/maxcor10/ftol1e-12를 유지한다. gtol은 원래 w의1e-6에 대응하도록 S*1e-6으로 변환한다. maxls만30으로 늘리되 실제 CUDA 전체평가≤60은 초기w0를 포함해 그대로다. 원래 E203의 maxls10과 비교하므로 두 solver 설정 변화가 합쳐진 시험임을 명시한다. 각각의 인과 효과를 분리해서 주장하지 않는다.

## 구현과 확인

새 `tools/fit_c_banc_m3_scaled_local_response.py`와 `tests/test_c_banc_scaled_local_response.py`만 실제 gpt-6-sol/high가 구현한다. 부모 FIRST → 필요시 같은 Sol 첫 수정 → 부모 SECOND/잔여 수정이다. E203의 PersistentLocalResponse, DecoderAdjoint, validate_response, objective_and_command_gradient, direct_metrics, feasible, load_targets와 E202 loader/IO를 읽기 전용 재사용한다. 독립 runner가 새 freeze/source review를 확인하며 기존 gate를 monkeypatch하거나 건너뛰지 않는다. CUDA 계산식·launch/block/options·장치 버퍼는 바꾸지 않는다.

계수/경계 변환 및 chain rule은 생성 quadratic와 실제 생성 decoder 목적함수에서 central FD로 검산한다. 합격 허용오차는 E203의 abs2e-5+rel2e-4다. 정확히 같은 w의 cache, 초기점 재사용, 실제 평가60회 상한, 오류 시 네 신경 배열과 최선 원자료 보존을 확인한다. 이미 검산한 CUDA 계산기를 수정하지 않으므로 불필요한 CUDA 시험 반복은 하지 않는다. 원본 첫w0는 실제 CUDA를 한 번 실행하여 E202 네 배열 bit exact, decoder atol2e-12/rtol1e-12 및 E203 원래 merit/gradient_w atol2e-12/rtol1e-12를 확인한다. 실패하면 optimizer0으로 종료하며 원자료를 보존한다.

원본 E203의 전체command20%개선/전체·각다리velocity 및 각다리command 비악화/startupq RMS.018/max.04/startupLTM RMS.009/max.025를 유지한다. penalty10000/innerguard.995/regularization1e-4*mean(w²)와 모든 sample mask도 동일하다. feasible 평가 중 command RMS 최소, 없으면 merit 최소, 동률 최초의 선택 규칙을 유지한다. 같은 계수에 대한 gradient와 값은 cache하며 cache key는 w의 float64 bytes다. 최대평가/iteration/선 탐색 종료가 유한하게 보존되면 제한된 탐색 결과로 판정한다. 계산/유한성/기준복원/시간/출력 실패는 invalid 또는 inconclusive이며 자동 재실행하지 않는다.

평가마다 free_weights(w), scaled_variables(u), gradient(gradient_w), scaled_gradient(gradient_u), 전체391×7 weights, command, metrics/terms/merit/시간/순서를 다음 평가 전에 저장한다. 선택한 네 신경 원자료/decoder/weights/motor IDs와 u를 최종 판정 전에 보존한다. 실패 시 확보된 최선 자료도 `unverified_incomplete_raw`로 남기고 원래 오류를 유지한다. 최종 실제 통과만 `supported_conditional_local_response_proposal`, 미달은 `valid_bounded_search_no_proposal`이며 어느 쪽도 채택·실제 보행 개선·F1–F8 완료가 아니다.

## 판정 후 방향과 자원

예측은 첫 선 탐색을 넘어 유한한 감소를 얻는 것이다. 그것만으로 H204의 보행 관련 목표를 통과하지 않는다. 조건부 기준을 모두 통과하면 실제 BANC의 재귀 전류와 감각·신체가 바뀌는 후속 시험을 설계한다. 미달이면 보존된 iterate/gradient/보호조건을 근거로 잔여 병목을 분석하고 같은 실패를 무조건 반복하지 않는다. 최선 E150, M2, 기존 실패와 F1–F8을 보존한다.

기존 누적5144.166472199704/5444초·잔여299.833527800296초에서 감독240초를 예약하며 배정을 늘리지 않는다. CPU1/GPU1/원본process1/optimizer1/CUDA≤60/추가warmup0/retry0, fit180/내부210/외부225/감독240초, 출력256MiB/GPUbuffer768MiB/free disk4GiB다. 최대 조건부600초/240만tick/960만stage이고 전체BANC/physics0이다. 준비·생성·부모 검산 비용을 별도로 기록한다. 미측정은0이 아니다. 기존 사용자 개발 권한 내 가역적인 solver 변경이며 UI/서버를 변경하지 않는다.
