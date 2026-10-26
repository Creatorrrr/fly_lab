# E167: E150 감쇠 전달 가족의 낙관적 오차 하한

2026-09-23, 부모 설계. E166은 대응 coverage가 부족해 여섯 다리 모두 inconclusive였다. 아래쪽 요청만으로 전달 고갈을 원인으로 선택하지 않는다. E107의 출력 파형 개선 가능성과 E164의 제한된 onset 가족 실패를 바탕으로, 새로운 시간 상태를 구현하기 전에 감쇠만으로 조건부 파형 제안 기준을 만족할 여지가 있는지 한 번 계산한다. 전체 M3 목표와 F1–F8은 변경하지 않는다.

## 질문과 고정 자료

E143의 signed-moment interval과 distance-to-interval 방법을 재사용한다. E143의 E142 pooling 결과 수치를 E150에 재사용하지 않는다. E150 native-recruitment-state, adapter-provenance, fixed-monotone-parameters, E164 pre-solve-raw 및 부모 final review를 고정 입력으로 사용한다. 원본 배열의 E167 계산은 설계와 해시 동결 후에만 한다. 새로운 state/신경/물리/기구학/적합은 없다.

시계는 2001행의 tick0..2000, 5ms다. 391 MN, 96근육, 42관절, LF/LM/LH/RF/RM/RH 각 7관절의 기존 순서다. E164 target_indices와 target을 그대로 사용하며 tick100..1923의1824개여야 한다. 새 phase 정렬·offset·target 재선택은 없다. frozen E164 검사와 원본 해시를 재사용하되, 현재 읽은 값의 dtype·shape·finite·ownership·완료 mask·clock을 검증한다. 숫자를 유효하게 보이도록 정수나 bool로 손실 변환하지 않는다.

## 가족과 계산

소유 MN i의 양수 기여는 c_i(t)=sum_k[low_i,k min(bank_i,k(t),kappa_i)+high_i,k max(bank_i,k(t)-kappa_i,0)]이다. bank·low·high·kappa의 비음수와 slope의 단일 소유 근육 지원을 확인한다. high-low 자체는 음수일 수 있다. E164 raw bank와 recruitment bank가 같아야 하고, variable77근육에서 sum_owned(c_i)와 실제 기록 activation의 최대 차이는 1e-12 이하여야 한다. fixed19의 activation은 original_activation과 비트 동일해야 한다. fixed 목록은 E164에서 그대로 복사한다.

검사 가족은 a'_m(t)=a_recorded_m(t)-sum_owned_i gamma_i(t)c_i(t), 0<=gamma_i(t)<=.5다. fixed19에서는 gamma=0이다. 이 가족은 제안 중인 R in [.5,1], lambda in [0,1] 감쇠식보다 느슨하다. 시간별 gamma를 독립 허용하며 branch floor·tonic budget·startup·속도 오차 제약도 모두 풀어준다. 실현 가능한 시간 상태를 찾는 최적화가 아니다.

amin=a_recorded-.5*C, amax=a_recorded이며 fixed19는 양 끝 모두 기록값 그대로다. 근사 .5*a_recorded로 바꾸거나 작은 음수를 clip하지 않는다. gain=.02와 frozen moment M, rest b를 사용해 qlo=b+.02(amin Mpos^T+amax Mneg^T), qhi=b+.02(amax Mpos^T+amin Mneg^T)를 계산한다. qbaseline=b+.02*a_recorded M^T와 E164 recorded_command의 tick1 이후 차이는 1e-12 이하여야 한다. t0는 원래 decoder 등식 대상이 아니다.

부동소수점에 대한 보수적 확장은 p(t,j)=max(1e-10,4096*eps64*(1+abs(b_j)+.02*((abs(amin)+abs(amax))@abs(M)^T)_tj))+.02*1e-12*sum_m abs(M_jm)이다. 최종 구간은 [qlo-p,qhi+p]다. 원래 끝점, 확장 끝점, p, activation 재구성 및 decoder residual을 먼저 저장한다. 이것은 정수/유리수 구간의 엄밀 증명이라고 부르지 않는다.

목표까지의 거리 d=max(qlo-p-target,target-qhi-p,0)를 원소별로 계산한다. 전체·각 다리·각 관절의 baseline RMS, d RMS 하한, target이 확장 구간 밖인 비율을 보고한다. 전체 하한이 .8*전체 baseline RMS+1e-9 rad보다 크면 `excluded_from_conditional_20pct_screen`이다. 그 밖은 `not_excluded_by_optimistic_envelope`이며 실현 가능한 후보/적합 성공을 의미하지 않는다. 다리별·관절별 통계로 전체 판정을 유리하게 바꾸지 않는다. 20%는 기존 조건부 파형 제안 screen일 뿐, 자연 보행의 필요조건이 아니다. 제외 결론은 이 E150 신호·감쇠 가족·screen에만 적용한다.

독립 관절 구간의 서로 다른 끝점이 동일 활성 벡터로 동시에 실현되지 않을 수 있다. 예를 들어 M=[[1,-1],[1,1]], a=[2,1], gain=.02, b=0이면 구간은 q0=[0,.03], q1=[.03,.06]이다. target=(0,.06)은 각 구간 안에 있지만 공동 실현되지 않는다. 이 예와 부호 반전/고정 근육/positive hinge/수치 확장/변조·실패 보존을 생성 자료로 확인한다.

## 구현과 실행 계약

부모 설계 → 실제 gpt-6-sol/medium FIRST 구현 → 부모 FIRST 리뷰 → 필요한 첫 수정은 동일 Sol → 부모 SECOND 리뷰와 잔여 직접 수정. 닫힌 수식과 기존 저장 runner를 쓰는 한정 진단이므로 medium을 선택한다. 소유 파일은 tools/analyze_c_banc_m3_attenuation_envelope.py와 tests/test_c_banc_attenuation_envelope.py다. 과거 소스/실험은 읽기 전용이다. 공통 파일 해시/atomic 저장 helper를 재사용할 수 있지만 E166 고유 snapshot/contract를 E167이라고 위장하지 않는다.

CLI --freeze/--freeze-sha256/--source-review/--source-review-sha256/--out. 승인 gate와 freeze/design/source snapshot을 실행 전후 검증한다. 필요한 원본 필드는 dtype/수치 assertion 전에 raw로 남긴다. 정렬 후 residual·구간·거리도 numerical assertion과 판정 전에 raw로 남긴다. 실패의 최초 오류와 partial/temp를 보존하고 늦은 실패 시 성공 result/manifest를 취소한다. 원본 실행 전에 무관 cwd/PYTHONPATH 없는 실제 script subprocess와 생성 full-run 오류 경로를 검사한다.

CPU1/GPU0, 원본 process1, 자동재시도0, fit/body/StateStore/GraphStore/neural/physics/Jacobian0. 내부40/외부60/감독70초, 출력96MiB, free1GiB. E166까지 실패 포함 보수누적2880.4512925000163초를 올림한2881+70=2951초 이내다. 이전 2953초 배정 내의 짧은 판별이다. 생성 테스트·보존 해시·부모 독립 검산은 따로 기록한다. M2/F6/UI를 변경하지 않으며 이번 진단으로 후보 채택이나 전체 목표 완료를 선언하지 않는다.
