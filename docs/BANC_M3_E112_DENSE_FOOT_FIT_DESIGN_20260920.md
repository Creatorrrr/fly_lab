# E112 고정 설계: BANC 운동 출력의 촘촘한 발 궤적·속도 보정

## 목적과 판정 경계

E110은 원래 개발 시작에서 F3 전진을 통과했으나 연속 전진 구간이 다리 한 주기보다 짧아 F4/F5가 실패했다. E107 외부 shape-only 진단은 같은 신체에서 연속 주기를 만들었다. E112는 **고정 BANC의 기존 운동뉴런 수신 가중치로 다리 안의 발 궤적과 전환 속도를 더 잘 만들 수 있는지** 한 번의 조건부 적합으로 검증한다. 앞선 E108/E109는 발 위치를 100ms마다 92개만 보정했다. 이번에는 실제 E110 입력과 전체 5ms 위치·인접 변화율을 사용한다.

이는 오프라인 조건부 계산이다. 최종 실행에 위상·교사·시간 궤적을 추가하지 않는다. 새 신경/근육 시간상수, 노드, 연결, rest, moment, pool, 신체 기구학, 힘줄 또는 root 보정은 없다. F1–F8 원래 목표와 이전 실패는 그대로다. 후보의 폐루프 보행은 이 계산으로 증명되지 않는다. F6 두 새 자세는 사용하지 않는다.

## 고정 입력과 실제 계산

부모가 노출된 E110 실제 2000개 관측 및 초기0에서 만든 `verification/banc-natural-gait-m3-20260919/parent-e112-training-inputs.npz`를 사용한다. SHA256은 `47ab37cb492afa2b6d5ca20dc5fbf0a8429167cea401c20c43aa17f7e89f999c`이다. 옆 JSON의 원본 source_paths/source_sha256와 원래 E110/E111 부모 리뷰를 확인한다. 기존 E108에서 재사용하는 것은 검증한 동일 해부학적 행렬/ID/분할과 원래 E100/E103 startup99 타깃이다. E105 source/current/phase/target은 재사용하지 않는다.

입력은 time 2001×5ms, source49, MN391, 기존 signed edge1572다. 실제 E110 float32 전류 합을 float64로 넓힌 current에서 `source_rate @ current_w.T`를 빼 고정 background를 만든다. E110의 38개 native tau 설정을 포함한 각 MN tau/a/threshold/cap와 96-muscle pool/moment/rest/LTM을 유지한다. 같은 전체0 초기 상태의 native conditional response는 기존 `linear_response(relu(cap*tanh(a*(current-threshold)/cap)), .005, tau)`, 10ms `endpoint_response`, `rest+.02*activation@moment.T`, `.2*tanh(activation_LTM/10)`다. MN adaptation은 기록에서0이며 새로운 상태나 감각 입력을 꾸미지 않는다.

목표는 E110 실제 각 다리 E1 위상에 같은 E85 다리 기준을 정렬한 원래 부모 분석값이다. 연속 index100..1927, time .5..9.635초의 **1828개** 명령과 위치를 사용한다. 발 속도는 인접한 두 위치 차이를 .005로 나눈 **1827개 구간 평균 속도**다. 구간 끝 연결, 빈 위상 건너뛰기, 반복 경계 차분은 없다.

검증된 E111 `geometry.npz`와 `TorchFootGeometry.leg`를 CUDA float64로 사용한다. 모델/context geometry hash는 `36a47d2f0f0d4e83d5ff8ecf1af95b9ce63269d5158d98642b4f89e0bc230953`이다. E112는 FlyBody나 전체 신경망을 생성하지 않고, 정적 기하학도 다시 추출하지 않는다. 1828개 위치를 GPU에서 계산하고 autograd는 인접 속도항까지 연결한다. 학습용 target/reference의 CPU→CUDA staging은 시작할 때 한 번 수행한다.

## 한 번의 최적화와 선택

LF, LM, LH, RF, RM, RH 순서로 기존 분할의 6개 독립 수신 가중치 블록을 최적화한다. MN 수69/63/63/70/63/63 및 기존 edge 분할을 검사한다. 각 블록은 현재 E110 actual float32 가중치에서 유도한 동일 efficacy의 log를 초기값으로 사용하는 **단일 SLSQP**다. 양수 efficacy 범위 .001..10, 원래 signed weight×efficacy, 부호/부재 연결/다른 행은 유지한다. 임의 restart·여러 초기값·tau 탐색은 없다.

다리별 목적함수는 다음 네 평균 제곱 항을 동일 가중치로 더하고 기존 작은 가중치 prior를 더한다.

1. 7개 관절 명령 오차 / `max(target joint std, .025rad)`.
2. 3D 발 위치 오차 / `max(target foot std, .05mm)`.
3. 인접 5ms 발 속도 오차 / `max(target velocity std, .5mm/s)`.
4. 해당 LTM 출력의 .2 대비 오차 / .2.

가중치 prior는 `(6e-5*n_edges/1572)*mean((efficacy-1)^2)`다. 정규화 값은 후보와 무관한 동결 타깃에서만 구한다. 위치·속도는 모든 해당 sample/interval의 평균이며 임의 다리 또는 시점을 제외하지 않는다.

원래 .005..495초의 99개 startup 제한은 qRMS .02rad/qmax .05rad/LTMRMS .01/LTMmax .03이다. 실제 E110과 과거 조건부 계산에서 관측한 초기 전이 오차를 고려해 이번 최적화/선택에서는 더 안쪽인 **.018/.04/.009/.025**를 각 다리에 요구한다. 이 여유는 새 후보의 실제 전이 오차를 보장하지 않는다. 원래 99개와 원래 제한도 별도 기록한다. 초기 E110은 원래 max .056618rad이며 새 제한에도 infeasible임을 반드시 표시하고, 이를 이유로 몰래 초기값/타깃/제한을 바꾸지 않는다.

SLSQP는 목적함수와 네 제곱 비율의 정확한 autograd를 사용한다. 원래4개 비율을 새 안쪽 제한으로 정규화하며 optimizer cap `(1-2e-4)^2`, 저장 후보 selector cap `(1-1e-4)^2`다. 동일 float64 log vector의 callback 중복은 cache로 한 번만 계산한다. 평가 시작 전에 비용을 기록하고 완료 결과·원래 optimizer 반환·기울기·네 제약·모든 log parameter를 남긴다. 각 다리의 finite best-any와 **안쪽 제한을 통과한 최저 목적함수**를 별도로 보존한다. 6개 다리에 모두 feasible 기록이 없으면 합친 후보가 없다. 제한 도달은 수렴으로 표시하지 않는다.

6개 후보를 합치면 원래 CSR에 적용할 **실제 float32** 가중치로 반올림하고 전체391MN 조건부 응답을 한 번 다시 검증한다. 기존 calibration의 `values` 중 선택1572개와 `individual_edge_efficacies`만 바꾼다. 나머지 모든 필드/38tau/선택 밖 weight/원래 edge signs/ID는 exact로 확인한다. 원래 legacy theta를 새로운 출력 제어기로 사용하지 않는다. calibration은 검증 전 후보이며 product 기본값이나 전체 neutral checkpoint를 만들지 않는다.

## 검증과 결과 분리

구현 전 합성 CPU 테스트로 인접 차분(상수 궤적·알려진 선형 궤적·뒤쪽/앞쪽 gradient), shape/nonfinite/연속 시간·빈 구간 거절, infeasible 초기값 선택 방지, 실제 float32 반올림 뒤 제한 재검사, 중복 callback·시작 비용과 오류 시 원자료 보존을 확인한다. 본 응답 자체/기하학의 기존 검증을 재작성하지 않는다. 부모 첫 리뷰→같은 Sol 첫 수정→부모 재리뷰/필요한 남은 직접수정 이후 첫 CUDA 적합을 별도 승인한다.

첫 적합에서는 동결 부모 NumPy baseline에 대한 CUDA response/activation/q/LTM/current 최대차를 기록하고 q/LTM/current≤1e-9, rate/activation≤1e-8을 요구한다. 모든 raw baseline을 검사 전에 보존한다. E110 실제에 대한 초기99 qRMS≤.002/max≤.01, 훈련 qRMS 재구성 오차/실제 타깃오차≤.1, 발 속도 RMS 재구성 오차/실제 타깃오차≤.1을 확인한다. 현재 발 proxy 속도 RMS .050811mm/s 대 실제 타깃 차이3.072840mm/s이며 최대오차4.732541mm/s는 별도 기록한다. E110 원래 startup 실패는 알려진 상태이며 baseline 자체를 feasible이라고 요구하지 않는다.

후보 출력은 원자료부터 저장하고 원래99 제한과 새 안쪽 제한, 모든 위치/속도/관절 RMS, 각 다리 값, LF 말단 오차를 보고한다. **새 밀집 발 가설**은 actual E110 대비 위치RMS≤.8, 속도RMS≤.8, 전체 및 각 다리 qRMS≤1.05와 새 startup 제한을 모두 통과할 때만 supported다. 기존 `native_shape_screen`(q/foot≤.8, LF말단≤.5, 각leg≤1.05)와 원래 E101 absolute caps도 수치/판정을 그대로 별도 계산한다. 새 가설 통과로 기존 실패를 덮지 않는다. `local_supported`는 새 가설·기존 상대/절대 shape·원래 startup·안쪽 startup 모두 통과해야 true다. `body_validated=false`, `adoption=not_applicable`, `F1_F8=not_assessed`를 유지한다.

부모는 실제 float32 예측·원본 source hash·기준·계수·비용을 독립 재계산해 리뷰한다. 후보 또는 실패를 받은 뒤에만 다음 실제 전체 BANC 신체 질문을 설계한다. E112 코드가 자동으로 신경/물리/F6를 실행해서는 안 된다.

## 자원과 구현 소유권

E111까지 실제 보수 공유 비용320.4130187초를 **321초**로 잇고, 새 밀집 발·속도 질문에 전체 외부 프로세스 **185초**를 배정해 공유 상한 **506초**로 기록한다. fit≤150초, 각 다리≤24초(저장 포함), 각 다리 신규 vector≤1000회, 합≤6000회, SLSQP maxiter1000/ftol1e-9. baseline1/final1은 별도 validation 호출로 기록하며 초기점 별도 trial은 없다. deadline 전 .5초를 저장 여유로 남긴다. 시작·완료 평가, 실패·부분 진행·정리·외부 import 비용을 구분하고 이전 비용을 삭제하지 않는다. 외부 supervisor가185초 한도를 적용한다. 오류/초과는 raw와 원 오류·비용을 보존하고 자동 반복하지 않는다.

원래 신경 advance0/물리 advance0이며 CUDA 소유자는 Sol 하나다. 부모는 동시에 source/data 리뷰만 한다. 구현·테스트는 기존 명시적 **gpt-5.6-sol xhigh**에 맡긴다. 1572개 제약 최적화, native response, GPU 기하학 미분 및 과학적 판정 경계를 함께 다루는 난도에 맞춘 설정이다. 새 agent 생성 한도 때문에 기존 Sol을 재사용한다. 쓰기 범위는 새 `tools/fit_c_banc_m3_dense_foot_recruitment.py`, `tests/test_c_banc_dense_foot_recruitment.py`, 이후 `verification/banc-natural-gait-m3-20260919/e112-dense-foot-recruitment/`뿐이다. 기존 evaluator/helper/E110/E111/M2/product와 부모의 동결 입력·문서는 읽기 전용이다.
