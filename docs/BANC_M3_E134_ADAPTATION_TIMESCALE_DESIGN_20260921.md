# E134: 근육별 적응 시간상수의 전체391 CUDA 계산 검증

2026-09-21 KST · 부모 설계 · 원본 실행 전 계약

E132의 LF 제약 적합은 원래 국소 목적함수를 4.6096% 줄였다. E133은 같은 점에서 5% box 반경의 선형화 개선 예측이 0.0330625%에 그침을 확인했다. 이것은 국소 결과이며 전역 불가능성의 증명이 아니다. 같은 gain/edge 적합의 반복보다, 지속 파형을 바꿀 수 있는 별도 동역학 자유도를 먼저 정확히 구현한다. E109의 membrane tau 배율 재시험은 하지 않는다.

기존 적응식은 z'=(g*r-z)/0.3이다. 새 연구 모델은 근육 j별 ell_j=log(T_j/0.3), T_j=0.3*exp(ell_j)를 사용한다. T 범위 [0.1,3]초는 공학적 탐색 범위이며 측정된 생물학적 값으로 주장하지 않는다. 초기 rate/adaptation/activation은 기존 기록 그대로 고정하며 파라미터의 함수로 바꾸지 않는다. 원래 membrane tau, gain 범위[0,8], 기존 edge provenance/입력 누적 순서, float32 네 단계 native 전류/source, float64 RK4 dt=.00025, 20tick/control, pool/moment/rest/LTM/EMA .01초는 유지한다. 외부 시계/위상/CPG/teacher/root 보정은 추가하지 않는다.

## 식과 구현 경계

r'=(phi-r)/tau_mem, phi=max(cap*tanh(a*(I-threshold-z)/cap),0).
gain 감도 v,w는 v'=(-phi_prime*w-v)/tau_mem, w'=(r+g*v-w)/T.
ell 감도 u,h는 u'=(-phi_prime*h-u)/tau_mem, h'=(g*u-h)/T-(g*r-z)/T.
기존 edge 감도 p,q는 p'=(phi_prime*(weight*source-q)-p)/tau_mem, q'=(g*p-q)/T.
네 RK4 단계 모두 해당 단계 r,z와 감도를 사용한다. 각 MN은 정확히 한 근육에 속하므로 자신의 gain/ell 감도만으로 전체 근육별 대각 감도를 표현한다. B 열 순서는 edge1572, gain96, ell96이며 각 열의 실제 근육 mapping을 보존한다. 이는 기록된 presynaptic 입력 아래 조건부 감도이고 전체 재귀 BANC 미분이 아니다.

새 tools/banc_m3_adaptation_timescale.py는 기존 입력 검증/소유 복사/전송 기록을 재사용할 수 있으나 자체 CUDA kernel/solve를 소유한다. 기존 E128/E130 소스나 전역 factory를 변경하지 않는다. E130처럼 delta의 expm1을 한 번 준비한다. log tau 전송 및 exp 준비, forward/decoder/bank의 요청·동기화·미확정을 구분하고 추가 감도 raw도 오류 시 회수한다. 실제 CUDA가 없으면 실패한다. ell=0의 기존 계산은 아래 수치 및 분기 계약을 만족해야 한다.

## 동결 입력과 순서

원본 E128 fixture와 E126에서 봉인한 E110/E113/E121 네 단계 전류/source를 재사용한다. 3문맥,391 MN,96근육,69source,2444 union,1572 fit edge,각10초다. 부모가 고정 파라미터 NPZ를 만든다: base delta/gain은 전체0에 E132 선택 LF 값만 원래 global mask 위치로 삽입한다. 이 결합은 수치 감사 입력이며 보행 후보가 아니다. center delta는 base와 같고 center gain_j=1+.25*sin(j), center ell_j=.25*sin(j+.5), edge 방향 d_i=.5+.5*cos(.31*i)다. 생성된 float64 bytes를 SHA로 봉인한다.

부모 최종 소스 리뷰 후 단일 CUDA 원본 프로세스에서 아래15 batch를 한 번씩 실행한다. 원본 shape/행/열/시간을 제외하지 않는다.

1. E128+E130 기존 계산: base delta/gain, derivative bank 포함.
2. 새 계산: 같은 base, ell=0, bank 포함.
3. 새 계산: center, bank 포함.
4. center ell에 h를 모든96근육에 동시에 +/-: h=1e-5,1e-6,1e-7,1e-8의8 batch.
5. center gain에 +/-1e-8:2 batch.
6. center delta에 +/-1e-8*d:2 batch.

총15 batch/45 context response/450조건부초/1,800,000 context RK4tick/703,800,000 scalar MNtick/90,000 decoder endpoint, derivative bank3회/2,263,680,000 edge-stage tick이다. 별도 synthetic tiny warmup은 기존1/새1 batch,각1MN/1문맥/20tick/.005초이며 원본 비용과 구분한다. 추가 신경/신체/geometry/최적화/F6 노출은0이다. 각 batch의 모든11 response field와 branch bits를 무압축 NPZ로 저장한 뒤 다음 batch를 시작한다. 기존 baseline은 원래8 field다. 세 bank는 모든 시간/열을 저장한다. 기술 오류 시 남은 batch를 중단하고 partial/raw/primary/secondary 오류 및 요청된 미확정 비용을 남기며 자동 재시도하지 않는다. 수치 불일치는 모든 예정 raw 보존 뒤 판정한다.

## 사전 고정 수치 판정

ell=0은 기존8 field 및 B의 기존1668열 max 절대차<=1e-8, branch bits exact여야 한다. center의 log-tau 분석 감도를 각 h의 중심차분과 비교한다. 최종 기준은 사전 고정 h=1e-8이며, r/z/activation/q/LTM 각각 max_abs_error<=2e-5+2e-4*max_abs(FD)다. 큰 h의 실패와 argmax/각 stage 분기 변화도 모두 보존한다. gain의 같은5 field와 edge 방향의 activation/q/LTM도 동일 h/오차 기준을 적용한다. 파라미터 순서/ID/dtype/shape 및 finite도 검사한다. ell=0 비교는 tau 신규 감도 자체를 0이라고 요구하지 않는다.

원본이 기술적으로 완전하게 기록되면 부모가 center의 r/z/gain감도/logtau감도/pool/EMA/q/LTM을 독립 NumPy float64 네 단계 RK4로 딱1 batch 계산한다. GPU의 모든11 field와 max차<=1e-8, 모든 branch exact여야 한다. edge 방향은 CUDA bank와 유한차분 및 소형 독립 각-column 시험으로 검증한다. 부모 참조에 원본 CUDA 코드/새 RHS 함수를 재사용하지 않는다.

H134는 새 시간상수 계산과 감도의 정확성이다. 통과는 이 계산기를 다음 제약 적합에 쓸 수 있다는 뜻이고 자연스러운 보행/전체 BANC/생물학적 기전의 검증은 아니다. 미통과하면 사후 기준 변경이나 유리한 h 선택 없이 실패를 보존한다.

## 자원와 분업

E133까지 주 프로세스 누적1516.4284439000185초 및 이전1530초 allocation과 모든 종료 계약은 변경하지 않는다. 새로운 H134에 대해서만 사전 누적 기준1517+CUDA supervisor150+부모 참조 supervisor75=1742초를 배정한다. 이는 이미 승인된 무예산 M3 안의 새 가설 계약이며 E132/E133 연장/재시도가 아니다. CUDA 내부120/외부145/supervisor150초, 부모 NumPy 내부55/외부70/supervisor75초. 원본 각1프로세스/재시도0/GPU1개/CPU BLAS thread1이다. 준비·합성·리뷰 비용도 별도 기록하며 기존 비용을0으로 초기화하지 않는다.

부모는 설계/동결 입력/독립 참조/원본 supervision/리뷰를 소유한다. 실제 gpt-5.6-sol/xhigh가 새 모듈, tools/verify_c_banc_m3_adaptation_timescale.py, tests/test_c_banc_adaptation_timescale.py만 구현한다. 새 비선형 동역학·CUDA 변분식·상태/예외 재현 때문에 xhigh를 사용한다. 부모 첫 리뷰→필요하면 같은 Sol 첫 수정→부모 두 번째 리뷰→잔여 부모 직접 수정 순서를 지킨다. Sol은 원본/fit/전체 신경망/UI를 실행하지 않고 작은 synthetic만 시험하며 모든 invocation/실패/비용을 보존한다.

소형 시험은 heterogeneous T/gain/초기 adaptation, 서로 다른 stage2/3, 여러 MN/근육 및 signed/multiple incoming edges, 모든 edge/gain/ell 감도, 기존 ell0 parity, 잘못된 ell 범위·shape·NaN, 소유 복사, 요청 직후/forward 동기화 뒤/decoder/bank/host/raw 저장 실패의 partial 및 primary/secondary 보존을 다룬다. 검증되지 않은 새 코드에서 원본 자동 실행은 금지한다.

통과하면 동일 초기99/shape20%/F1–F8을 유지하는 시간상수 포함 제약 적합을 별도 동결한다. 조건부 통과 후 native full158706의 per-cell T와 snapshot/restore/CUDA graph identity 및 같은 후보의 실제 신체 검증이 필요하다. 이 단계를 건너뛰고 GUI나 기본 모델에 적용하지 않는다. M2와 활성 UI, 미사용 F6 자세, 과거 실패를 보존한다.
