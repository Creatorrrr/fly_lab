# E210: 고정 인과 기저 적합 결과

원본 CPU 프로세스1회와 다리별 Clarabel6회가 완료됐지만, 완성된 조건부 후보는 없다. LF/LH/RF는 `AlmostSolved`, LM/RM/RH는 `PrimalInfeasible`다. 고정한 `Solved` 조건을 그대로 적용해 beta 합성·2×2 대조·native 통합을 실행하지 않았다. 오류 대신 반환된 미해결 결과와 원시 해/슬랙/dual/목적함수/모든 solver 입력을 보존했다.

| 다리 | 반환 | 반환 시간(초) | 비고 |
| --- | --- | ---: | --- |
| LF | AlmostSolved | 21.410350 | 완전 수렴 기준 미달 |
| LM | PrimalInfeasible | 6.702282 | 이번 추가 제약 포함 문제의 수치 certificate 확인 |
| LH | AlmostSolved | 20.657100 | 완전 수렴 기준 미달 |
| RF | AlmostSolved | 21.550451 | 완전 수렴 기준 미달 |
| RM | PrimalInfeasible | 8.114838 | 이번 추가 제약 포함 문제의 수치 certificate 확인 |
| RH | PrimalInfeasible | 10.450519 | 이번 추가 제약 포함 문제의 수치 certificate 확인 |

부모는 원래 D/moment/표본에서 물리 행렬을 재구성해 목적 P/선형항/상수와 startup, 전체 cone 행렬을 독립 대조했다. command/velocity norm의 Gram 상대오차는 최대2.537e-14다. LM/RM/RH의 dual cone 위반0, A^Tz 최대3.848e-10, b^Tz 각각-14.19713/-7.42769/-3.91918로 이번 제한 문제의 수치 불가능 certificate를 확인했다. 이를 원래 전역 조건이나 자연 보행 자체의 불가능 증거로 확대하지 않는다.

특히 E210은 계산 분해를 위해 원래 전신 startup RMS .018보다 엄격한 각 다리 startup RMS .018을 추가했다. 전신 조건이 허용하는 다리별 필요조건 .018*sqrt(6)를 대입하면 기존 certificate는 더 이상 음수가 아니다. 이것은 더 넓은 문제의 가능성을 증명하지 않지만, E210 결과가 원래 전신 제약의 불가능을 증명하지 못함을 확인한다. 후속은 저장된 여섯 물리 문제를 결합해 하나의 원래 전신 startup norm을 정확하게 적용하는 사전 설계다. 최종 proxy/F1–F8 기준을 완화하지 않는다.

구현은 부모 설계→실제 gpt-6-sol/high→부모 FIRST→동일 Sol 첫수정→부모 SECOND·잔여1건 순서다. 모의9개/Ruff와 별도 생성 실제 Clarabel1회가 통과했다. 부모 잔여 수정은 solve 직전 ledger 저장 실패를 실제 호출로 세지 않도록 회수하는 처리다. 생성 실제 해 [.7999999952,4.6558e-9]는 독립 경계해 [.8,0]과 일치했다.

원본 감독99.8154093초, 내부98.2947367초, solve 호출합88.0444684초, 최대 프로세스트리 RSS1,840,209,920byte, 출력226,621,388byte, 남은 프로세스0/retry0다. Clarabel 설정20초는 반복 경계에서 확인되는 soft limit여서 세 반환은20초를 소폭 초과했다. 설정 상향/재시도 없이 합120·내부180·외부220초를 지켰으며 각 호출이 정확히20초에 멈췄다고 주장하지 않는다. 생성 wrapper1.1981512초/실제solve.0002179초와 부모 원자료 검산6.6086655초는 별도다. 준비1.2119044초 및 테스트/리뷰도 원본과 구분하며 미측정 비용은0이 아니다.

보수적 원본 누적5616.236763699704/5784초, 잔여167.763236300296초다. 새 CUDA·전체BANC·신체0이며 E150/M2/UI/F6·실패 이력을 보존했다. F1–F8은 active·미완료다.

근거: `verification/banc-natural-gait-m3-20260919/e210-causal-neural-basis-fit/`, `parent-e210-source-review.json`, `parent-e210-output-review.json`, `parent-e210-global-bound-diagnostic.json`, `parent-e210-final-review.json`.
