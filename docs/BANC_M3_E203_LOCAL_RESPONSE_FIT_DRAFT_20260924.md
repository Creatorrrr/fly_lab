# E203 초안: 국소 MN 반응 학습

이 문서는 E202 검증과 병행한 부모 설계 초안이다. 아직 실행 계약으로 동결하지 않았고, E202의 원래 10초 복원과 독립 검산이 통과해야 구현·실행 계약을 확정한다. F1–F8과 미달 기록은 그대로다.

## 질문과 비교

기존 신경 출력을 그대로 두고 근육 전달을 바꾼 여러 가족은 고정 명령 파형 조건을 만족하지 못했다. E203은 E202의 기존 r/z 기반 국소 비선형 응답을 학습하면, 해부학적 MN→근육 전달과 초기 자세를 유지하면서 참조 명령 오차를 줄일 수 있는지 묻는다. 전체 회로의 입력은 E201에 조건화되어 있으므로, 통과하더라도 실제 BANC 폐루프 보행의 제안만 된다.

입력은 E201의 실제 160000×391 stage current와 생리·초기 상태, E150의 원래 decoder다. 목표와 판정 mask는 기존 E197 pre-solve raw에 보존된 E85 own-leg/offset 목표, E103 초기99 목표를 그대로 재사용한다. target_indices 100..1923의1824행, 인접 velocity1823쌍, startup1..99다. baseline은 원래 E150의 기록이며 새 계산기의 출력으로 바꾸지 않는다. 교사·시각·위상은 적합 목적에만 사용한다.

고정 여섯 LTM 근육을 담당하는 MN의 w는 전부0이다. 나머지 MN의7계수는 [-1,1]에서 학습한다. 무출력 MN을 강제 활성화하지 않으며, initial r/z, threshold, coefficient, gain, inverse tau, 연결 가중치, decoder slopes와 모든 초기 상태는 고정한다. singleton 근육의 신경 반응은 학습 대상에 포함하지만 그 근육 전달은 원래 그대로다. mask는 graph index/MN root/근육 소유권으로 실행 전에 저장한다.

## 계산 경로

E202의 검산된 CUDA kernel source·식·float64/비융합 옵션을 그대로 재사용한다. 새 장치 객체가 current와 실제 생리 및 출력 buffer를 한 번 할당하고, 각 평가에서 w만 바꾼다. 모든 평가는 같은 고정 초기 r/z와 접선0에서 시작한다. 최초 w=0 평가를 평가 횟수에 포함하여 E202 raw와 비교한다. 원본의 숨은 warmup이나 별도 baseline 재실행은 없다.

보정식과 7개 접선은 E202 그대로다. 저장 rate에서 원래 causal EMA 세 bank와 10ms pooled activation을 계산하고, 원래 고정19/가변77 decoder와 .02 moment·rest를 적용한다. CPU의 매 control Python 근육 loop 대신 검산할 수 있는 vectorized filter를 사용할 수 있다. 원래 함수와 생성자료 및 w=0 E202 raw에서 수치 일치를 요구한다. 경계 bank=κ에서 hinge 미분은0으로 정의한다.

역전파는 command gradient→.02 moment 전치→근육 gradient→동일 소유권의 low/high slope→각 EMA의 역방향 adjoint→rate gradient 순서다. fixed19는 원래 pooled 10ms EMA로 역전파한다. 마지막으로 시간축에서 rate gradient와 E202의 국소 rate_sensitivity를 수축하여 MN별7계수 gradient를 얻는다. 전체 42×2737 Jacobian을 구성하지 않는다. 다른 MN의 재귀 전류 변화는 이 조건부 미분에 포함되지 않는다.

## 학습 목적과 제한

아래 수치는 초안이며 원본 계산 전에 최종 동결한다. 제안은 L-BFGS-B 1회, 초기w0, seed/random restart0, 최대60개의 실제 CUDA 목적·미분 평가다. 별도 평가 없이 동일w cache를 재사용하고, 원본 호출 횟수를 정확히 기록한다. 현재 평가 결과를 보존하기 전에 다음 평가를 시작하지 않는다.

기본 목적은 전체 command MSE를 (.8×원래 baseline command RMS)^2로 나눈 값이다. 보호 항목은 전체 velocity, 각 다리 command/velocity, startup RMS와 startup 각 표본 절대오차다. 보호 norm 한도의 내부 목표는 원래 한도의 .995배다. 양의 정규화 MSE 초과량에 대한 제곱 벌점10000, 모든 startup 표본의 제곱 초과 벌점 합10000, 작은 평균 w² 벌점1e-4를 더한다. 20% 개선 판정 자체나 물리·시작 조건은 완화하지 않는다. 후보 선택은 사전 고정된 단일 merit 최소 평가이며, 미달은 이 제한된 탐색의 미달로 남기고 가족 전체 불가능성으로 주장하지 않는다.

실제 조건은 기존과 동일하다: 전체 command RMS≤.8 baseline, 전체/각 다리 velocity 및 각 다리 command 비악화, startup q RMS≤.018/max≤.04, startup tendon RMS≤.009/max≤.025. 6 LTM w0와 원래 decoder/hash/신경 상태·방정식/범위를 확인한다. 사후 clipping·정규화·다른 후보 조합은 없다. 학습 목적함수 감소는 이 조건들을 대신하지 않는다.

## 검증·자원·후속

부모 설계→actual gpt-6-sol/high 구현→부모 FIRST→필요시 같은 Sol 첫 수정→부모 SECOND/잔여 수정. 작은 생성자료에서 filter adjoint와 전체 loss gradient를 central FD 및 E202 NumPy 경로로 검산한다. 시간 순서·prefix·initial row0·fixed19/LTM mask·평가 상한·실패 raw 보존을 검사한다. 원본 실행은 소스와 입력을 동결한 뒤 부모가 한 번만 수행한다.

예상 예약은 단일 CPU/GPU, 내부270/외부285/감독300초, 최적화1회/평가최대60/자동재시도0, 출력256MiB/GPU buffer768MiB/free disk4GiB다. E202 후 실제 잔여 시간을 확인한 다음 기존5444초 배정 안에서 가능한지 확정한다. 현재 이 초안은 예약이나 배정 증가가 아니다. 원본 process·합성 테스트·부모 검산 비용을 구분하며 미측정을0으로 쓰지 않는다.

최선 평가의 실제 w·목적·gradient·명령·상태, 평가별w/지표·호출/시간/종료 원인, 원본 비교·hash/source를 보존한다. 제안이 통과하면 동일 국소 방정식을 전체 BANC의 실제 stage dynamics에 넣고 outgoing feedback·원래 감각·신체를 포함한 짧은 검증부터 설계한다. 실패하면 실제 학습 잔차와 active constraint를 검토한다. F6는 노출하지 않고 M2/UI/최선E150을 보존한다.
