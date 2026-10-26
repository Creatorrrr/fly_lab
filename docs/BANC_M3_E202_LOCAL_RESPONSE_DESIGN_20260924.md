# E202: 학습 가능한 국소 MN 반응과 원본 복원

E201에서 동일 E150의 실제 160000단계 입력을 확보했다. 이번에는 이 입력으로 391 MN을 계산하는 조건부 모델을 구현한다. 이는 학습용 계산기이며 전체 회로와 신체를 대체하지 않는다. 새 계수의 적합·선택과 제품 통합은 이번 범위에 없다. 고정 F1–F8, E150의 F4/F5 실패, 모든 과거 실패, F6 미노출, M2/UI 보존을 유지한다.

## 새 모델 가정

각 MN의 원래 상태 r(발화율), z(적응)를 유지한다. I는 E201이 기록한 drive+recurrent current다. 기존 coefficient c=a/cap, threshold θ, cap C, membrane tau τ, adaptation gain g, 실제 native inverse adaptation tau k를 사용한다.

```
s = c*(I-θ-z)
v = tanh(s); x = r/C; y = tanh(z/C)
f = [1, v, v*v, v*v*v, x, x*x, y]
o = sum(w[j]*f[j], j=0..6)
δ = 0.5*tanh(o)
h = s+δ
φ = C*max(tanh(h),0)
dr/dt = (φ-r)/τ
dz/dt = k*(g*r-z)
```

w는 MN마다 7개이며 모두 유한하고 [-1,1] 범위다. 0.5의 logit 보정폭과 기저함수는 명시적인 공학 가설이며 실측 생리 상수가 아니다. w=0이면 원래 방정식으로 돌아간다. 목표 발화율 φ는 비음수이며 C에서 포화한다. 외부 시각·위상·참조 궤적을 세포 입력에 추가하지 않는다. 향후 실제 네트워크에서는 변경된 r이 동일 BANC의 기존 outgoing 연결에 되먹임되어야 한다. 저장된 I를 런타임에 주입해서는 안 된다.

새 내부 상태는 추가하지 않지만 기존 자신의 r/z와 입력에 따른 비선형 응답을 갖는다. E123/E128/E134의 gain/T 조정, E148/E164/E175/E197의 발화 후 전달 변경과 다른 가설이다. 모든 MN의 활성화나 특정 위상을 목표로 삼지 않는다. 학습할 w의 범위는 실제 적합 전에 별도로 고정한다.

## 계산과 미분

실제 네 RK stage I를 순서대로 사용하여 dt=.00025의 RK4를 계산하고, 20 tick마다 총 2001개 경계를 출력한다. 초기 r/z는 E201 기록에서 한 번만 읽는다. 이후 관측 rate를 잔차 보정이나 teacher forcing으로 더하지 않는다. native float32에서 유래한 c/g/k 등의 실제 값을 그대로 float64로 승격한다. k를 다른 T 식으로 다시 계산하지 않는다. 이 미분은 고정 I에 대한 조건부 미분이며 전체 회로의 미분이 아니다.

7개 계수의 국소 접선 R_j=∂r/∂w_j, Z_j=∂z/∂w_j를 동일 RK4 stage에서 적분한다. 초기 접선은 0이고, h<=0에서 정류 함수의 미분은 0으로 정의한다.

```
f_r = [0,0,0,0,1/C,2*x/C,0]
f_z = [0, -(1-v*v)*c, -2*v*(1-v*v)*c,
       -3*v*v*(1-v*v)*c, 0,0,(1-y*y)/C]
p = 0.5*(1-tanh(o)^2)
φ_r = C*(1-tanh(h)^2)*p*dot(w,f_r)  (h>0)
φ_z = C*(1-tanh(h)^2)*(-c+p*dot(w,f_z)) (h>0)
φ_wj = C*(1-tanh(h)^2)*p*f[j] (h>0)
dR_j/dt = (φ_r*R_j+φ_z*Z_j+φ_wj-R_j)/τ
dZ_j/dt = k*(g*R_j-Z_j)
```

NumPy 참조와 CuPy float64 kernel을 각각 구현한다. CUDA는 MN마다 한 thread에서 시계열을 순회하면서 r/z와 7개 접선을 유지하고 경계만 출력한다. 원래 158706개 노드를 줄인 실행 모델이라고 부르지 않는다. 원본 검증에서 CPU fallback은 금지한다. 원본 실행은 w=0의 forward+접선 한 번이며 최적화는 0회다. current를 GPU에 한 번 전송하고 native tick마다 CPU와 동기화하지 않는다. 반환 배열은 rate/adaptation 각각 (2001,391), rate_sensitivity/adaptation_sensitivity 각각 (2001,391,7)이다. 모든 배열의 유한값·shape·clock을 검사한다.

원래 decoder는 E150의 초기 bank, original.activation, 세 tau, kappa, slopes, 고정 19근육, pool/moment/rest/LTM을 유지하며 r로부터 2000번 업데이트하여 command/tendon을 만든다. 초기 row 0의 held command는 재구성 오차에서 제외한다. 외부 교사의 command로 대체하지 않는다. 전체 42×2737 계수의 거대한 Jacobian 대신 국소 MN 접선을 후속 학습에 전달한다.

## 검증과 판정

actual gpt-6-sol/high가 `tools/banc_m3_local_response.py`, `tools/verify_c_banc_m3_local_response.py`, `tests/test_c_banc_local_response.py`를 구현한다. 부모 FIRST→필요하면 동일 Sol의 첫 수정→부모 SECOND/잔여 수정을 적용한다.

작은 합성 배열로 0/nonzero w, 음/양 입력, 정류 경계, 계수 범위, 순서를 검사한다. NumPy/CUDA 상태·접선의 허용오차는 abs1e-9+rel1e-8, 경계를 피한 central FD는 eps1e-5에서 abs2e-5+rel2e-4다. 짧은 합성 시계열과 소수 MN의 7개 방향 전부를 확인한다. 다른 MN 계수의 영향이 없는 국소 모델임을 검사한다. 단순 상수 입력에서 w=0/정류 0/기존 rate 감쇠의 닫힌 식을 부모가 독립적으로 검산한다.

원본은 E201의 전체 160000×391 입력, w=0, 동일 초기 상태로 한 번 실행한다. 저장 native rate에 대한 RMS<=.5Hz/max<=5Hz, 96 activation RMS<=.25/max<=2.5, 42 command RMS<=.001rad/max<=.01rad, 여섯 LTM RMS<=.0005/max<=.005를 E126의 기존 fidelity 계약에서 유지한다. 이는 최대 허용값이며 실제 오차를 모두 보고한다. 전체 raw를 먼저 저장한 뒤 판정한다. 실패는 보행 가설의 반증과 구분하여 계산기/기준 복원 미달로 기록한다. 모든 접선이 유한하고 원본 계수가 0이며 입력 hash가 일치해야 한다. 학습 개선 여부는 판정하지 않는다.

## 자원과 다음 단계

기존 누적 5105.233614699708/5444초, 잔여 338.7663853002923초에서 **원본 process1/CPU1/GPU1/내부90/외부105/감독120초**를 예약한다. 상한 증가와 자동 retry는 없다. 출력 256MiB, 계산기 GPU buffer 768MiB 이내, 사전 free disk 4GiB다. kernel compile에 쓰는 작은 합성 warmup은 따로 계상하고, 원본 입력의 숨은 warmup/추가 fit을 금지한다. 원본 조건부 계산 10초/40000 tick/160000 stage와 실제 전체 BANC·physics 0을 구분한다. 부모 NumPy 검산은 별도로 측정하며 미측정 리뷰 비용을 0으로 기입하지 않는다.

E201 부모 최종 리뷰·manifest, E150 start checkpoint, 구현 SHA를 고정한 후 부모가 원본을 한 번 실행한다. 테스트에는 작은 생성 데이터만 사용한다. 성공해도 전체 회로/물리/F1–F8 검증은 미완료다. 성공 후 구체적인 학습 mask/목적함수/제약/탐색량을 별도 고정하고, 이를 통과한 후보를 전체 BANC와 실제 신체에서 검증한다.

