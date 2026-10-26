# E127: native 단계 입력의 CUDA 적응 응답과 전체391 감도

2026-09-21 KST · 부모 설계 · 원본 실행 전 계약

이 단계는 M3 F1–F8의 보행 개선을 위해 근육 적응·기존 연결 효능을 제약 적합할 계산 경로를 완성한다. E126은 E110/E113/E121 각10초의 실제391 MN 전류/69 source를 투명하게 수집했고, 단계 전류를 이용한 독립 gain0 재구성이 고정 fidelity 기준을 모두 통과했다. E124의5ms PWL 전류 오차와 큰 유한차분의 정류 경계 문제를 제거하지 않은 채 적합을 시작하지 않는다. E125의 선택11MN 결과를 전체391에 일반화하지 않는다.

## 변경과 유지

새 `tools/banc_m3_dense_adaptive.py`는 E124의 검증된 CUDA r/z/v/w·pool·EMA·q·LTM 구조를 재사용하되 입력을 실제 네 native RK4 단계의 전류로 바꾼다. E124/E125/E126 소스와 모든 기존 성공·실패 기록은 수정하지 않는다. 원래391 MN 순서, 실제 tau/a/threshold/cap, 초기 r/z/activation,96근육 pool/moment/rest/ltm,dt=.00025,20tick/control=.005,adaptation tau=.3,EMA tau=.01을 유지한다. 외부 위상·CPG·신체 보정·teacher를 실제 보행에 추가하지 않는다.

고정 입력 아래 식은 dr=(max(cap*tanh(a*(I-threshold-z)/cap),0)-r)/tau, dz=(g*r-z)/.3이다. 자신의 근육 gain에 대한 v=dr/dg,w=dz/dg를 같은 네 단계에서 적분한다: dv=(-phi_prime*w-v)/tau,dw=(r+g*v-w)/.3. phi_prime은 정류 양수 분기에서 a*(1-tanh²), 그 외0이다. 각 MN은 원래 pool에서 정확히 한 근육에 속한다. gain은96개/범위[0,8]이다. 이는 고정된 presynaptic 입력 아래의 조건부 감도이며 전체 재귀 BANC의 미분이 아니다.

Native stage2와 stage3 입력은 같은 중간 시각이라도 서로 다를 수 있으므로 PWL 보간하거나 합치지 않는다. 배열은 각 문맥 `[160000,391]` float32를 GPU에 한 번씩 올리고 RHS에서 float64로 변환한다. r/z/v/w·decoder는 float64다. CUDA가 실제로 없으면 실패하며 CPU fallback은 없다. 작은 생리·decoder 입력은 소유 복사로 고정한다. 고정 업로드/후속 gain 업로드 count·bytes와 실제 private nonblocking stream 동기화를 기록한다. 읽은 CPU 원본을 이후 수정해도 GPU 응답이 달라지지 않아야 한다.

각 native tick의 네 양수 분기를 4bit로 묶은 uint8 `[contexts,neural_ticks,neurons]`를 r/z/v/w와 같이 저장한다. 이것은 미분 오차의 분기 차이를 설명하기 위한 기록이다. 원래 수치 식의 tanh 판단을 사용하고 별도 재계산·반올림으로 분기를 정하지 않는다. 전체391을 모두 포함하며 사후에 불리한 열/시각을 제외하지 않는다.

## 원본 비교와 호출 수

부모가 E126 세 current .npy,initial/control endpoint,독립 dense reference,원래 E124 fixture와 기존 source/manifest에서 입력을 동결한다. 신경·신체·GraphStore·StateStore를 새로 실행하지 않는다. 모든 문맥은 이미 노출된 E110/E113/E121이고 F6 두 자세는 미노출로 남긴다.

부모 최종 소스 리뷰 뒤 단일 CUDA 프로세스에서 다음10 batch를 정해진 순서대로 한 번씩 실행한다. 각 batch는 세 문맥 각10초,120000 context RK4tick,6000 decoder 업데이트다.

1. gain0.
2. 중심 gain: 근육 index j=0..95에 대해 `1+.25*sin(j)`를 부모가 float64 배열로 저장한 값. 모든 근육을 같은 값으로 두어 mapping 오류를 가리지 않는다.
3. h=1e-5의 중심+h,중심-h.
4. h=1e-6의 중심+h,중심-h.
5. h=1e-7의 중심+h,중심-h.
6. h=1e-8의 중심+h,중심-h.

총10 CUDA batch/30 context response/300 conditional초/1200000 contextticks/60000 decoder 업데이트다. 가중치 변경·최적화·실제 신경/신체 진행은0이다. 별도 tiny CUDA warmup 한 번의 실제 비용을 main과 분리한다. 기술 오류면 남은 batch를 시작하지 않고 partial output과 requested/completed/unknown CUDA 비용을 보존한다. 각 batch raw 저장 후 다음 batch를 시작하며, 수치 판정은 모든10개의 raw를 보존한 뒤 수행한다. 테스트/시험 실패도 숨기거나 자동 재시도하지 않는다.

Gain0은 E126의 actual 끝점과 비교하여 rate RMS/max .5/5Hz,activation .25/2.5Hz,q .001/.01rad,LTM .0005/.005를 그대로 적용한다. E126 NumPy dense gain0와의 float64 비교도 모든 rate/z/activation/q/LTM에 대해 max1e-8로 검사한다. 원래 초기99 q RMS.02/max.05,LTM RMS.01/max.03에 대한 실제/조건부 오차를 별도로 보존하며 E110 실패를 제거하지 않는다.

각 h에서 전체 r/z와96근육 activation,42q,6LTM의 중심차분을 분석 감도와 비교한다. 기존 E124의 field별 `max_abs_error <= 2e-5 + 2e-4*max_abs(FD)` 기준을 유지한다. 원래 모든 문맥/시각/열을 사용하며, 최종 판정은 사전 고정한 가장 작은 h=1e-8의 모든 field 통과다. 다른 h 결과·argmax 위치·각 문맥의 분기 변경 tick/bit 수를 함께 보존한다. 더 큰 h의 통과를 대신 선택하지 않는다. 이를 자연스러운 보행 또는 원래 E124 실패의 번복으로 부르지 않는다.

부모는 GPU 원본이 기술적으로 완전하게 기록된 뒤 중심 gain만 독립 NumPy float64 r/z/v/w로 한 번 계산한다.30 conditional초/120000 contextticks/6000 decoder업데이트이며, 모든8 output field를 GPU 중심과 max1e-8로 비교한다. Native stage table은 일반 연속 I(t)가 아니므로 원본에 DOP853 보간 가정을 추가하지 않는다. 작은 연속/constant synthetic에서의 검증과 이미 검증된 E124 식을 재사용하고, 원본 discrete RK4는 네 입력을 직접 사용하는 독립 계산으로 확인한다. 이 부모 참조까지 통과해야 전체391 감도 구현을 다음 적합에 사용할 수 있다.

## 자원·실패·분업

단일 RTX GPU와 실제 측정 CPU/BLAS/Torch thread1을 사용한다. 원본 CUDA 프로세스 내부90초/최종보존15초,외부120초/supervisor125초다. 부모 참조는 내부35초/외부45초/supervisor50초다. E126까지의 기존 주 프로세스 누적1186.122813초를 보존하고 반올림1187+125+50=1362초로 배정해 기존 공유1530초 안에 둔다. 이전 준비/리뷰/합성 및 NumPy 비용도 따로 기록하며0으로 지우지 않는다. 원본 추가 신경·신체 진행0,fit0,F6노출0,새서비스/구매/UI조작0이다. 원본 raw와 source hash,각 요청/완료/미확정 비용,오류·후속 저장 실패를 보존한다.

소형 fixture는 다양한 gain/초기값/생리,중간 stage2≠stage3,겹치지 않는 근육 mapping,CPU/CUDA 모든 field 및 branch 일치,입력/출력 ownership,실패의 partial/state/cost 보존을 검사한다. 원본3문맥의 실행을 합성 테스트에 숨기지 않는다. 용도에 맞는 크기 검사를 적용해 차원·nonfinite·음수 생리·잘못된membership를 할당 전에 거부한다. 최대3문맥/160000stage/391MN/96muscle라는 원본 상한과 더 작은 합성 사례를 지원한다.

Sol/gpt-5.6-sol/xhigh가 새 `tools/banc_m3_dense_adaptive.py`, `tools/verify_c_banc_m3_dense_adaptive.py`, `tests/test_c_banc_dense_adaptive.py`만 구현한다. 부모는 실제 입력 준비·감도식의 독립 참조·원본 supervision·리뷰를 소유한다. 부모 첫 리뷰→필요하면 같은 Sol 첫 수정→부모 재리뷰→잔여 직접 수정 순서다. Sol은 synthetic만 실행하고 원본 또는 fit을 실행하지 않는다. 수정 전후 소스와 실패를 보존하며 재위임하지 않는다.

모든 fidelity/감도/독립 참조가 통과하면69 source 및2444 문맥별 기존edge 차이를 사용해 공동1572edge 효능·96근육 적응의 초기99/형태20% 제약 적합을 이어서 설계한다. 실패면 이유를 기록해 동일 F1–F8을 향한 다음 유효한 조치를 선택한다. 조건부 입력 및 현재 구현의 검증만으로 실제 보행 후보나 전체 목표를 완료 처리하지 않는다.
