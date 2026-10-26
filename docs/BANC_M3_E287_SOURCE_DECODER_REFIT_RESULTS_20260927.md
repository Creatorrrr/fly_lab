# E287 — 전체 source+decoder 적합 기각, E288 신체 실행 미개시

2026-09-27. 실제 gpt-6-sol/xhigh 구현→root FIRST→같은 Sol 첫 수정→root SECOND를 완료했다. 최종94개 생성검사/Ruff와 독립48변수 수식·원자료 목적함수·실제 엔진 참조 해제 검산을 통과했다. E288도 Sol/high 첫 수정 뒤61개 검사/Ruff 및 root의 저장 전체 초기 상태·완전주기 경계 검사를 통과했다. 이는 구현 검증이며 보행 성공은 아니다.

단일 원본 E287 실행은 유효했지만 후보를 만들지 못했다. Startup101/teacher2001행을 각각 한 번 CUDA 계산하여 원래 E277 q/LTM를 재현했고, 두 context에서 총42000 forward ticks·derivative0·동시에 engine1개였다. 전체330 free MN/77 variable muscles,3960변수의 단일 Clarabel이15회 반복 후 `PrimalInfeasible`을 반환했다. Candidate0, development 평가0, 실제 신체0이다. 학습20%·다리별 개발 기준을 통과한 것으로 취급하지 않는다.

Root가 저장된 문제와 solver dual을 별도 검사했다. 두 dual cone margin은 .002628166997/.003324360682로 양수이고, 모든 slope/envelope가 [0,2]에 있는 원래 feasible domain에서 weighted slack의 최대값 상한은 −.005164623589였다. 보수적인 수치 오차 여유2.1397006e−6을 더해도 음수다. 따라서 **고정 E277 source 응답과 이번 startup/envelope 제약 아래의 전체 affine decoder 가족**에 한정해 제약 충돌을 뒷받침한다. 이는 전체 신경 동역학이나 자연 보행의 불가능성 증명이 아니다. Dual 질량은 LM peak행에 가장 크게 놓였으나 그 수치만으로 생물학적 원인을 확정하지 않는다.

원본 supervisor5.341421200021초, 현재 process-tree RSS peak1791979520bytes, 프로세스 자체의 보고된 working-set peak1936523264bytes였다. 추가 GPU 메모리 상한 검사는3508369120bytes의 engine 추정에 기반하며 실제 측정 GPU peak라고 부르지 않는다. 출력43622929bytes, 원본 PID17544/46296은 종료했다. 시작 전 RTX4080 free13502MiB, 다른 FlyLab Python0, disk160238379008bytes를 확인했다. 원본 누적은7869.382029599514/7925초, 잔여55.617970400486초, 예약0이다. Root 저장값 감사.0531291초 및 dual 감사.0228766초는 추가 원본/solver 실행 없이 수행했다.

E288의 소스와 조건부 실행 준비 도구는 남겼지만 선택된 후보가 없으므로 freeze·원본 예약·신체 실행은 하지 않았다. 같은 decoder 재적합·RF threshold/radius/solver 반복으로 돌아가지 않는다. 다음 구조 변경은 기존 감각 전달과 정적/저역통과 출력 보정에서 벗어나 기존 premotor 재귀회로 자체의 활동 의존적 동역학을 다룬다. 이 결정은 개발 방향이며 아직 양성 행동 증거가 아니다. M2/E150/활성 UI, F1–F8, 모든 과거 기각, 미노출 F6를 보존한다.

증거: `verification/banc-natural-gait-m3-20260919/parent-e287-original-final-review.json`, `parent-e287-saved-ray-audit.json`, `parent-e287-output-manifest.json`, `parent-e287-final-source-review.json`, `parent-e288-final-source-review.json`. Source review 동안 발생한 root 생성 harness 오류 두 건과 Sol generated fixture/partial-receipt 오류도 각각의 FIRST/SECOND 기록과 비용에 남겼다.
