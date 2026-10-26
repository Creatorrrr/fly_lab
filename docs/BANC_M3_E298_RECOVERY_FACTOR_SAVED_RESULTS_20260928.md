# E298 결과 — 저장 상태에서 회복 전달식은 비자명, 보행 미검증

E296의 동일한 저장 rate/R에서 `g=min(1,R*(1+tau*u*r))`를 계산했다. 활성1513 source의 controls1..100 중103158/151300 endpoint(68.1811%)는 float32 계수1이며,48142 endpoint/1077 source에서 억제가 나타났다. 최소 계수는 .5762503147이다. 원래 u=.05/tau=.150/보호 roster를 변경하지 않았다.

초기 controls1..10(.005–.050초)은 전부계수1이다. Controls11..20에서는173 source/879 endpoint가 억제되고,21..100에서는1077 source/47263 endpoint가 억제된다. 이는 E296의 이미 저장된 상태에 적용한 대수 결과다. 새 모델의 초기 출력 exact, 새로운 궤도에서 같은 비율, startup 개선, 주기·전진·생물학적 보행을 입증하지 않는다.

| 고정 구간 | 계수1 비율(활성 source만) | 새 source 손실 / 기존 source 손실 | 새 선택 양수 target 손실 / 기존 손실 |
|---|---:|---:|---:|
| 1..10 | 100% | 0% | 0% |
| 11..20 | 94.1904% | .17557% | .20485% |
| 21..100 | 60.9526% | 6.78885% | 6.11743% |
| 1..100 | 68.1811% | 6.16396% | 5.57241% |

손실은 같은 unmasked rate에서 `(1-float64(factor32))*rate64*weight64`로 구한 모델 입력 단위의 양수 연결 합이다. 모든7136 source와7136 target의101행을 보존했다. 중간 masks는 관측되지 않았으므로 실제 CUDA 전류라고 하지 않는다. 비활성5623 source의R/g1, 모든 core향6172선택 연결 계수1, 음수 선택 연결의u0가 확인됐다. 양수 target 손실이0보다 크므로 계수만 비자명하고 rate0인 퇴화 사례도 아니다. g≥R의 국소 관계가 관절 오차 개선을 보장하지 않는다.

Actual gpt-6-sol/high FIRST 구현 후 root는 두 신규 소스 전체를 리뷰하고26tests/Ruff/독립 스칼라 손계산을 통과시켰다. 수정 필요가 없어 R1을 만들지 않았다. 실제145개 입력파일/37개 소스 SHA와 기존 E296 producer lineage의 passive gate를 통과한 뒤 저장 실행1회가 완료됐다. Root는 별도 SciPy sparse 계산으로 모든7136 target×101행을 검산했고 최대 차이0, source arrays exact, 고정구간190개 공개 통계를 재확인했다. 테스트/감사 PASS는 보행 PASS가 아니다.

비용은 agent 생성7.6612507초(실패0), root 생성 감독6.472502899996471초, passive gate 프로세스4.0947165초(내부1.0839714), 저장 감독7.099515100009739초/관측 tree peak1689309184bytes(자식 자체 peak1855844352bytes), root raw감사 프로세스1.4246011초(내부.7775354), 통계감사 프로세스.6695853초(내부.3163756)다. Original/model/body/CUDA/solver/fit 전부0이며 원본 누적8078.0529733995245/8150초, 예약0이다. 모든 소유 프로세스 종료는 별도 CP 영수증으로 확인한다.

근거 파일은 `verification/banc-natural-gait-m3-20260919/e298-recovery-factor-saved/`와 `parent-e298-saved-audit.json`, `parent-e298-summary-audit.json`이다. Freeze SHA `7f7a39b905c0e9c350f98732a63672bbed7f1abbce39b41c1a5e5d669786c7ec`, review SHA `1aff7c5e6c603a2b0045872a1bdc9854e41fe756905003d8b00614d968baaf5d`. Runner SHA `40c2f57d0c6e42aaa7fcd2e9a9bd2cf7f3b9b189c875e16c7073dfdfcb8895be`, test SHA `7561403ea486469d47b0791a24c6a798fc02b7d1deced458d9089586d17f3d36`다.

Root 판정: 이 계산은 일정 전달을 유지하면서 상태에 따라 억제가 남는 별도 신경식의 구현·tiny 자격검사를 진행할 근거가 된다. 후보 채택은0이고 다음 원본 신체 실행은 아직 예약하지 않는다. 미래 같은-stage CUDA 계산과 unity 원래식 분기, 음수 RK단계 검출, whole 저장·복원은 별도 설계한다. F1–F8/M2/E150/UI/F6 미노출 및 모든 과거 기각·비용을 보존한다.
