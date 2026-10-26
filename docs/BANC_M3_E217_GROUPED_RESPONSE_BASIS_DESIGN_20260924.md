# E217: 근육별 공유 계수와 64개 상태 반응 기저

E216은 기존 gate/cap/막시간상수/10ms 전달로 전신 conditional proxy를 만족하는 시간별 forcing이 있음을 확인했다. E214의16기저 반응식은 같은 조건에서 불가능했다. 따라서 다음 가족은 생리 출력 상한을 확대하지 않고 상태에 따른 반응 해상도를 높인다. 64기저를 모든 MN에서 독립 적합하면 dense 학습행렬과 parameter 수가4배가 되므로, **동일 근육에 속한 MN의 64계수를 공유**하는 별도 가족을 사용한다. 이 가족은 기존16기저 가족의 엄격한 상위집합이 아니며 성공을 미리 가정하지 않는다. E217은 이 새 가족의 정확한 조건부 기저 구축/검산만 한다. 학습·native 통합·신체 검증은 통과 뒤 별도 설계다.

## 외부 시간 입력 없는 반응 함수

E213의 per-MN scale, x=(log1p E/sE, log1p I/sI, asinh drive/sD), h_fast16ms 및 실제 inverse_adaptation_tau를 사용하는 h_slow, descriptor9=[x,h_fast,h_slow], fast-only gate, cap/tau/r0와 RK4 .00025초/20tick당 .005초 control을 그대로 사용한다. runtime 각 MN은 자기 현재 입력과 자기 기억만 읽는다. 새로운 신경 edge나 다른 MN의 실시간 상태, phase/clock/teacher를 반응 입력에 추가하지 않는다. 파라미터 공유와 아래 prototype 선택은 offline 학습 구조다. 실제 BANC 생리식으로 검증되었다고 주장하지 않는다.

각 원래 근육 g의 MN집합 M_g, weight w_m=pool[g,m]*cap_m/sum_own(pool*cap)를 사용한다. 저장 E213 descriptor의 시점 t와c 사이 group 거리²는 `sum_m w_m*sum_a(descriptor[t,m,a]-descriptor[c,m,a])²`다. 2001개 control 상태에서 row0을 첫 anchor로 삼고 farthest-point sampling으로64개의 공통 시점을 고른다. 매 단계 최근접 anchor 거리의 최댓값을 선택하며 동률은 최초 시점, 이미 선택한 시점은 제외한다. 모든 거리가0인 퇴화 그룹에서도 가장 이른 미선택 행을 사용해64개 고유 row를 남긴다. 목표 궤적/오차/시간값은 거리나 runtime feature에 넣지 않는다.

각 MN m의 실제 center[j]는 **그 근육의 동일 anchor[j] 시점에 기록된 m 자신의9D descriptor**다. 서로 다른 MN descriptor를 runtime에 합치지 않는다. sigma²_m은64개 own center의 엄격히 양수인 nearest-other-center 거리²들의 median과1e-4 중 큰 값, 전부퇴화하면1이다. 정확히중복된 center의 nearest 거리0은 median에서 제외한다. 이는 기존 E213 sigma 규칙을64개 center에 적용한 것이다. 사전 추가 bandwidth 배율이나 원본 sweep은 없다. 모든96근육/391MN의 geometry를 저장하지만 free90근육/343MN만 학습 대상이며 fixed6LTM/48MN은 기존 기록 발화와 전달을 그대로 사용한다.

`p_m,j=softmax(-||descriptor_m-center_m,j||²/(2sigma²_m))`, `phi_m=cap_m*gate_m*sum_j(beta_owner(m),j*p_m,j)`, beta∈[0,1]. h 상태는 기록 입력에만 의존하고 beta와 독립이다. neuron r'=(phi-r)/tau. 전원별 B'_m,j=(gate*p_m,j-B_m,j)/tau, B0=0과 H'_m=-H_m/tau,H0=r0를 기록하면 r_m=H_m+cap_m*sum_j beta_owner(m),j B_m,j다. 기존 H/H_factor와 초기조건을 바꾸지 않는다. 진단 probe는 모든근육 beta_j=(j+1)/65로 고정하며 kernel에서 직접 적분한 rate와 기저 선형합을 대조한다.

## 계산·저장 계약

이미 검산된 E213 history/descriptor를 prototype 선택에 재사용해 별도 history CUDA를 실행하지 않는다. E208 3stage-input 배열은 GPU에 각각 한 번 upload하고, 64기저 CUDA **basis kernel 한 번**으로2001×391×64 B 및 H/H_factor/probe_rate/h_fast/h_slow/descriptor를 기록한다. CUDA float64, fast_math/fmad 금지, 기존 .016 값을 basis64 상수 치환으로 바꾸는 실수를 금지한다. C loop/array/output index/probe denominator는 명시적 K=64로 일관되게 구현한다. 조건부 kernel은 전체BANC/물리 실행이 아니다.

canonical 전원 pool와10ms EMA로 추가 근육 기저를 직접 합한다: `D_group[t,g,j]=sum_m:owner=g pool[g,m]*cap_m*EMA10ms(B[:,m,j])`. fixed6근육 D_group은0이다. shape는2001×96×64, free90×64=5760개의 후속 학습 계수다. per-MN D 전체를 다시 저장하지 않는다. q=q0+.02*moment*sum_j(D_group*beta), q0/기존48MN/tendon/heldrow0는 E213 intercept를 재사용한다. 원본고정 E150의 actual rate를 canonical로 읽은 기준 결과도 보존하며 이것을 새 후보 성과로 보고하지 않는다.

E213와 같은 group별 sum_j D envelope, H/H_factor, h_fast/h_slow/descriptor를 수치 검산한다. h/H의 atol2e-12/rtol2e-12, 합산 D의atol2e-10/rtol2e-12, direct probe rate atol2e-9/rtol2e-11, direct/affine q atol2e-10/rtol2e-12를 사전에 사용한다. GPU output 비유한 값/음수 B<-1e-12, 원래 anatomy/parameter mismatch, invocation>1은 실패다. raw는 판정 전에 저장한다. probe 및 bounded beta의 재구성은 수학적 정확성 검사이며 원래 q/startup proxy 개선이 아니다.

source/hash/환경 및 input-transfer/kernel count를 ledger에 보존하고 group anchor indices, neuron center/sigma, weights/IDs/owner/free/fixed, 기존 parameters, B/H/probe/history, pooled affine map/q0/tendon, 비교 오차를 저장한다. 어느 것도 runtime 제어기로 채택하지 않는다. 후속 적합의 원본 호출/명령/패키지/SHA는 다음 단계에서 별도 고정한다.

## 구현·검증·자원

부모 설계→기존 실제 gpt-6-sol/high 구현→부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND/잔여수정. 소유는 tools/banc_m3_grouped_response_basis.py, tools/prepare_c_banc_m3_grouped_response_basis.py, tests/test_c_banc_grouped_response_basis.py다. 기존 E209/E213/E215/E216 소스·결과는 고정한다. 부모는 FPS거리/동률/상태 필터/pooled affine를 독립 계산한다.

순수/모의 검사 후 작은 생성 입력의 실제 CUDA basis1회만 허용한다(소수 MN/40tick 이하/외부30초). 64센터/공유계수/probe/두기억의 numpy 독립 대조와 실패회계 확인이 목적이다. 원본은 한 프로세스/새 CUDA basis1회/history0/fit0/전체BANC0/body0/retry0. CPU1/GPU1, GPU소유2GiB/RSS5GiB/출력768MiB/free4GiB, 내부90/외부110/감독120초다. 실제 CUDA 장치와 남은 메모리, 재사용 package/source/input identity를 직전 확인한다.

누적5696.177954499701/6004초, 잔여307.8220455002993초 중120초 예약, 추가 배정0. 준비/생성/부모검산/agent 비용은 별도이며 미측정을0으로 기록하지 않는다. F1–F8 active·미완료/F6미노출/최선E150/M2/UI 보존. 학습이나 native/body 실행은 이번 허용량에 포함되지 않으며 자동으로 연결하지 않는다.
