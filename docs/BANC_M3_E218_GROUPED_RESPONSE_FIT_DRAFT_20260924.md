# E218 후보 설계 초안: 근육별 공유64 반응계수 적합

E217의 부모 독립 검산이 통과한 뒤에만 구체화할 초안이다. 이 문서는 실행 허용이나 성공 증거가 아니다. E217 결과 SHA·실제 잔여예산·소스 리뷰를 포함한 별도 freeze 전에는 원본 solver를 호출하지 않는다.

목적은 시간별 자유 입력으로 E216에서 확인한 여유를, 자기 신경 입력과 두 causal 기억만 사용하는 고정 반응 함수로 실현할 수 있는지 확인하는 것이다. 같은 근육의 MN은64개 계수를 공유하고 free90근육만 학습하므로 변수5760개다. 고정6LTM·48MN 및 기존 q0/tendon은 유지한다. 기존16개 독립 MN 반응계수 가족의 상위집합은 아니다.

E217의 `D_group[t,g,j]`에 원래 moment를 곱한 `J[t,k,g,j]=.02*moment[k,g]*D_group[t,g,j]`를 사용한다. LF/LM/LH/RF/RM/RH 자유근육수16/14/15/16/14/15, 변수수1024/896/960/1024/896/960을 원래 anatomy에서 검증한다. q=q0+J*beta, beta[0,1]. 원래 E197 target·velocity·startup 행과 전역/다리별 proxy를 유지한다. E214와 동일한 다리별 정규화 command 목적함수와 작은 ridge, command/velocity 비악화 norm, startup max, 전역 startup RMS를 사용한다. 원래 전역 command20%개선과 모든 raw proxy는 결과에서 별도 평가한다. 수치보호 PAD를 물리 기준으로 대체하거나 beta를 clip하지 않는다.

가능하면 이미 검산된 quadratic/SOC 압축과 전역 조립을 재사용하되 shape를 억지로 MN16 형태로 위장하지 않는다. group/64 shape의 명시적 builder와 저장형식을 사용한다. 각 행렬/offset/column ownership·압축 오차·원래 물리 단위의 제약을 부모가 별도로 검산한다. 원본 solver는 최대1회, 원본 neural/CUDA/body는0이다. 성공한 raw 계수는 같은 고정 입력에서 direct 원래 pool/10ms 전달과 affine q가 일치하고 엄격한 기존 proxy를 통과해야 다음 단계의 후보가 된다.

실패는 이 고정 반응 가족과 입력/전달 조건의 실패로만 기록한다. 미실행 또는 제한 초과는 생리적 불가능성으로 해석하지 않는다. infeasible 판정은 raw dual cone과 원래 unpadded 제약을 독립 검사하며, E215에서 수정한 근사 cone의 잘못된 증명 문제를 반복하지 않는다. native/BANC 피드백·실제 몸체 성과로 확대해 해석하지 않는다.

부모 설계→기존 실제 gpt-6-sol/high 구현→부모 FIRST→필요시 같은Sol 첫수정→부모 SECOND/잔여수정이다. 구체 자원 한도는 E217 이후 잔여예산에서 사전에 고정한다. F1–F8·최선E150·F6미노출·M2/UI 보존을 유지한다.
