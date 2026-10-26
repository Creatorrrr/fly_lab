# E211: 전신 출발 제약 결합 결과

원래 전신 startup norm을 적용한 단일 결합 문제도 `PrimalInfeasible`로 종료했다. 조건부 후보를 만들지 않았으며 native 통합·2×2 대조·신체 실행·UI 채택은0이다. E210의 다리별 추가 startup RMS 제한만이 미달 원인이라는 설명은 충분하지 않다. 같은 고정 기저의 학습을 그대로 반복하지 않는다.

부모가 독립 재구성한34451×5488 sparse A, block 목적 P, 원시 startup matrix는 저장 solver 입력과 exact였다. A nnz10,767,527/P nnz617,784,19cones다. 원본 반환은11iter, solver 보고14.7371682초(설정/준비 일부 포함), 실제 solve 호출13.8156692초, 감독19.7866914초다. 원본1/solver1/retry0, 코드·원시 저장 오류0이다.

독립 검산에서 raw x/s/z가 유한하며 dual cone 위반0, A^Tz 최대4.7202e-9, b^Tz=-14.16299778이다. 원래 패딩 없는 q/velocity/점별 startup/전신 startup 수치를 저장 certificate에 대입해도 b^Tz=-14.13396073, beta0..1에서 A^Tz의 최소-6.7567e-8로 수치 분리가 유지됐다. 이는 검산된 **고정 조건부 기저·압축 제약 문제**의 미달 근거다. 실제 자연 보행 전체나 모든 신경 반응 모델의 불가능 증명이 아니며, 조건부 proxy가 최종 보행의 필요조건이라는 주장도 아니다.

구현은 부모 설계→gpt-6-sol/high→FIRST→동일Sol 첫수정→SECOND/부모 잔여수정을 따랐다. sparse 결합의 좌표·cone 순서·목적상수, raw 저장/잘못된 shape/1회 호출/실행 전 자원 차단, 원래 합격 수치와 PAD 진단 분리를 검증했다. 모의10개/Ruff가 통과했다. 180초 solver 제한은 soft, 내부240·외부270·감독300초는 별도이며 이번에는 초과0이다.

생성 수학 대조의 첫 실행은 Solved이나 좌표atol1e-5 검사에서 실패했다. 이를 보존하고 [사전 수정 기록](BANC_M3_E211_GENERATED_CHECK_CORRECTION_20260924.md)에 따라 정확한 KKT 항등식과 목적/제약tol1e-8로 검사식을 고쳤다. 별도 생성1회 추가 후 목적오차1.9948e-9, 제약잔차-1.5614e-9, KKT 항등식·분석해 범위가 통과했고, 부모도 저장된 primal/dual을 독립 검산했다. 생성총2회이며 원본재시도0이다. 첫 생성의 전체 raw가 assertion 전에 출력되지 않았다는 한계를 숨기지 않는다. 모델/원본 solver/최종 수치를 바꾼 것은 아니다.

메모리 최대 개별 프로세스 working set2,184,130,560byte, 감독 표본의 트리 RSS최대2,093,817,856byte다. 두 수치는 측정 방식이 달라 서로 대신하지 않는다. 출력144,335,091byte, 남은 프로세스0이다. 원본 누적5636.023455099708/6004초, 잔여367.976544900292초. 준비1.2331581초, 생성실패 wrapper1.3358127초/수정 wrapper1.211421초, 부모 원자료 검산2.9517218초/생성 independent.0003863초/패딩 제거 certificate.0777472초는 별도다. 기타 미측정 리뷰 비용은0이 아니다.

후속은 저장 기저에서 가능한 관절 명령의 낙관적 범위를 한 번 계산해 출력 범위 부족과 시간 패턴 제한을 구분하는 읽기 진단이다. 원본 신경·신체를 다시 실행하거나 해를 만들지 않는다. [native 통합 메모](BANC_M3_NATIVE_CAUSAL_RESPONSE_INTEGRATION_DRAFT_20260924.md)는 후보 검증 이후의 초안으로 유지한다. M2/최선E150/UI/F6·실패 이력과 F1–F8 active·미완료 상태를 보존한다.

근거: `verification/banc-natural-gait-m3-20260919/e211-global-startup-fit/`, `parent-e211-output-review.json`, `parent-e211-unpadded-certificate.json`, `parent-e211-final-review.json`.
