# E124: 근육별 활동 적응의 CUDA 조건부 응답과 감도 검증

2026-09-21 KST · 부모 설계 · 실제 응답 실행 전 계약

전체 M3 목표와 F1–F8은 유지한다. E123은 gain0 동등성, gain1 신경 효과, 신경·근육 상태의 저장 복원을 검증했다. E121 실제 보행은 F3만 통과했고 F4/F5는 미달이다. 본 단계는 적합에 필요한 계산을 실제 관측으로 검증하며, 보행 후보 채택이나 생물학적 기전 검증을 뜻하지 않는다.

## 설계 결정

기존 ContextResponse에는 적응 상태가 없다. r/z와 gain 감도를 동시에 적분하는 별도 도구를 추가한다. 입력 전류는 각 원래 문맥 E110/E113/E121에서 기록한 MN recurrent_current + drive다. 0초는 실제 StateStore 신경 상태와 drive를 확인하고, 전체 rate가 0인 경우에만 recurrent current 0을 사용한다. 기록한 발화율 잔차를 응답에 더하지 않는다.

E121이 줄인 억제성 port 중 83877은 기존 관측 source55 목록에 없다. 따라서 그 기록만으로 E121 가중치를 E110/E113 문맥에 정확하게 이식했다고 주장할 수 없다. 본 검증은 세 문맥 각각의 실제 전류를 보존한다. 향후 적합은 이 입력 한계를 해결하거나 후보/문맥 계약을 명시한 별도 설계를 거친다. 기존 두 문맥·형태 판정·초기99를 폐기하지 않는다. 이 변경은 초안의 조건부 적합 전 자료 검증을 구체화한 것이다.

## 계산 계약

- float64 입력, CUDA float64 연산. 실제 native float32 오차와 기록의 5ms 표본화 오차를 별도 fidelity 지표로 남긴다. 자동 CPU fallback은 없다. CPU 구현은 독립 소형 테스트에만 사용한다.
- 각 문맥의 2001 시점 0..10초, 관측 간격 .005초. 실제 float32 tau/a/threshold/cap을 float64로 확장한다. 적응 τ=.3초, 근육 EMA τ=.01초, native 미세 간격 .00025초로 20 RK4 하위 단계.
- 전류 I(t)는 각 5ms 끝점 사이 선형 보간한다. RK4의 각 단계에서 해당 실제 시각의 전류를 사용한다. derivative는 동일 단계의 r/z로 평가한다.
- `dr=(max(cap*tanh(a*(I-threshold-z)/cap),0)-r)/tau`; `dz=(g*r-z)/.3`.
- `dv=(-phi_prime*w-v)/tau`; `dw=(r+g*v-w)/.3`, v=dr/dg, w=dz/dg. 정류 경계의 phi_prime은 0, 양수 영역은 a*(1-tanh(...)²). 초기 v/w는 0으로 고정하며 실제 초기 상태는 gain과 독립이다.
- 각 MN은 기존 pool의 정확히 한 근육에 속한다. 96 muscle gain의 [0,8] 범위를 확인하고 391 MN에 배정한다. 각 MN별 하나의 감도만 필요하며 전체 재귀 회로 기울기로 부르지 않는다.
- 끝점 MN rate를 pool로 합산한 뒤 실제 초기 activation에서 EMA를 한 번 갱신한다. 초기 activation 감도=0. q=rest+.02*moment@activation, tendon=.2*tanh(activation[ltm]/10). q/tendon Jacobian은 activation 감도와 고정 moment/LTM 계수로 복원할 수 있는 compact factor로 저장한다. t0 실제 body command를 rest로 대체하지 않는다. t0 fidelity에서 제외하고 원래99는 .005.. .495초다.
- CUDA는 고정 입력을 한 번 전송하고 MN/문맥 별 병렬 시간 적분 kernel을 사용한다. 각 평가의 호출/요청 시간/완료 상태/partial 원자료를 기록한다. 초기 CUDA compile/warmup은 별도 소형 synthetic으로 계수한다.

## 실제 실행 순서와 판정

원본은 부모 최종 소스 리뷰 뒤 한 프로세스에서만 수행한다. 세 문맥을 동시에 평가하는 batch 순서는 gain96=0, gain96=1, gain96=1+1e-4, gain96=1-1e-4로 고정한다. 총4 batch /12 문맥 응답 /120 조건부 모형초 /480000 문맥 RK4 ticks. 실제 BANC neural/physics/fit/LP/F6 노출은 모두0. gain1은 진단값이며 선택된 보행 후보가 아니다.

gain0 fidelity는 각 문맥에서 t>0 전체2000과 원래99를 각각 보고, 전체2000에서 다음 RMS/max 기준을 모두 만족해야 한다: rate .5/5 Hz, activation .25/2.5 Hz, q .001/.01 rad, LTM .0005/.005. 이는 원래 보행 판정의 대체 기준이 아니라 표본 전류 재구성의 공학적 정확도 제한이다. 모든 원자료를 먼저 저장한 후 판정하며 실패 시 임계값이나 보간을 바꾸어 재실행하지 않는다.

gain1의 감도는 ±1e-4 중앙차분과 각 필드의 모든 근육 동시 변화 방향에서 비교한다. rate, adaptation, activation, q, tendon별 최대 절대오차 <= `2e-5 + 2e-4*max(abs(FD))`. q의 방향감도는 activation 감도@moment.T에서 계산하고 LTM은 tanh 미분을 사용한다. gain0는 경계여서 이 중앙차분의 중심으로 사용하지 않는다. 비선형 정류 경계로 미분 검증이 실패해도 raw와 최대 위치를 보존하고 정류 평활화를 사후 적용하지 않는다.

gain1과 gain0의 실제 기록 전류 아래 변화를 원래99 및 .5..10초에서 따로 보고한다. 이 차이는 전체 재귀/감각/신체를 다시 실행한 결과가 아니다. 같은 이득의 shape/startup 적합 가능성 및 F3–F5를 통과했다는 결론은 내리지 않는다.

단위 검증은 zero input, constant/nonconstant current, 비영 초기 r/z/activation, gain0·positive, 입력/ID/시간/pool 검증, CPU–CUDA, 독립 DOP853, finite difference, batch 독립성, CUDA 오류 시 자동 fallback 없음을 포함한다. 기존 신경/신체를 소형 테스트에 생성하지 않는다. 각 테스트 실행의 소형 적분과 참조 적분 수, 실제 CUDA 사용/skip, wall을 보고한다.

## 자료와 자원

부모가 실제 세 문맥/StateStore에서 `e124-inputs.npz`를 준비한다. 각 문맥의 actual rate/z/activation/q/LTM, actual initial state, current와 motor ID/physiology/pool/moment/rest/LTM/membership를 담는다. 원본 파일과 새 fixture, 실제 재사용 소스, E123 최종 리뷰를 SHA256으로 `e124-frozen-design.json`에 고정한다. E121/E122/E123 원본 manifest와 M2 source 목록을 보존한다. 원자료·ledger·stdout/stderr·exit code·외부 wall·소스 snapshot을 새 디렉터리에 남긴다. 기존 실패/결과 파일을 덮어쓰지 않는다.

원본 내부 상한90초(마지막15초는 보존용), 부모 프로세스 외부 상한120초, 자동 재시도0. CPU/BLAS/Torch 스레드1, GPU1개. 이전 실제 누적939.0747094999533초를 보존하고 보수적 이전값940초에서 본 단계 supervisor 포함 최대125초, 누적1065초. E123 자체988초 계약은 수정하지 않는다. fixture 준비(상한30초), 소형 테스트, 부모 독립 수치 재검토 비용은 별도로 정확히 기록하고 실제 신경/물리 시간과 섞지 않는다. 부모 독립 검토에 허용하는 신규 계산은 3 문맥×첫10구간×391MN CPU RK4 600 문맥 ticks, 5 필드 중앙차분/통계/decoder 대수 검산이며 원본 재평가가 아니다.

## 분업과 다음 결정

부모 설계 → Sol/gpt-5.6-sol/xhigh 구현 → 부모 첫 리뷰 → 같은 Sol 첫 수정 → 부모 재리뷰 → 잔여 직접 수정. 부모는 병행하여 원자료/해부학적 매핑을 준비·확인하고 원본 실행을 소유한다. Sol은 새 solver/runner/tests만 편집하고 원본 응답·신경·신체·적합은 실행하지 않는다.

재구성·감도가 유효하면 다음 단계에서 적응 gain의 제약 적합을 설계한다. 불일치하면 실제 native tick 전류 관측을 보강하는 방향을 검토한다. 어떤 경우에도 같은 최종 후보의 F1–F8 전체 검증이 완료되기 전에는 M3를 완료하거나 채택하지 않는다.
