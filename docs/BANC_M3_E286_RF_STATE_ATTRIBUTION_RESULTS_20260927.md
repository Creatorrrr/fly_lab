# E286 결과 — 고정된 정체점의 출력 부족을 확인하고 RF 세분화를 종료

2026-09-27. 두 실제 CUDA 조건부 상태를 재현한 실행은 유효하며 H286을 지지한다. 원래 neutral와 E285 마지막 계수의 q/LTM 101행, 계수·목표·오차 관계를 확인했다. Root는 저장 rate로 필터를 다시 계산하고, 다른 식 `low*min(bank,kappa)+high*max(bank-kappa,0)`으로 근육 및 관절 출력을 검산했다. 상태 재현·독립 decoder·명시적 motor ID·RK4 stage 창이 모두 통과했다.

E285 plateau의 control99에서 MN593 bank는 [4.0193e-6,5.9677e-6,.0089309231], MN141493은 [4.3251e-9,6.4218e-9,.0001885672]이며 모두 kappa=1보다 작다. 두 뉴런의 rate는 각각3.3549258e-6,3.6101665e-9로 **작은 양수**다. 마지막20ticks×4stage에서 두 뉴런의 활성 분기는 모두0회였다. 양의 방향 전체 관절 기여는4.5715886e-11 rad, 음의 방향 기여는−.0007307468 rad이며 rest=.049864를 더한 q=.0491332532, 목표=.0860724801이다.

이 결과는 해당 고정 상태에서 높은 전달 분기가 열리지 않았다는 진단이다. 전체 비선형 계수 가족의 불가능성이나 실제 지속 보행 실패의 원인을 증명하지 않는다. Control61/83에서도 양의 방향 출력은 작지만 요구되는 반대 근육 출력은 서로 달라, 한 시점의 gain만 올려 해결된다고 추정하지 않는다.

Actual gpt-6-sol/high 구현 뒤 root FIRST 수정 요구0. Sol 최종37tests/Ruff와 root37tests/Ruff 및 독립 생성 검산을 통과했다. 생성 검산은 순서가 다른 ID, high branch의 양측, fixed muscle, 작은 양수 보존,20×4 stage 창, 손상 자료 거부를 확인했다. 초기 생성 실패와 비용도 보존했다. Sol pytest measured fixture 합13.4378561초 및 추가 fake 진단1.8688847초, root 감독7.6619857초다. 모두 원본/solver/CUDA0이다.

원본2engine/2평가/forward4000/derivative0/solver0, supervisor2.334770799964초/RSS473919488bytes/출력20495885bytes. PID29944·40072 종료. 누적7864.040608399493/7905초, 잔여40.959391600507초이며 원본 작업·예약0이다. 원본 직전 RTX4080 free13608MiB/다른 FlyLab Python0을 확인했다. 독립 저장 검산 .1293705초는 별도다. F1–F8/M2/E150/UI/F6/이전 실패와 E280 자원 qualification을 유지한다.

RF startup 여유의 세분화는 여기서 종료한다. .036/.0162는 실제 .04/.018보다 강화한 내부 방법상 기준이며, 이를 못 맞춘 사실은 좋은 보행의 불가능성을 뜻하지 않는다. E279의 실제 전달 차이와 E277의 별도 full-fit/개발 실패를 함께 고려해 다음 묶음은 **변경된 source 출력에 맞춘 전체 운동 전달 재보정 → 제한된 full-fit → 동일 후보의 실제 startup/1.5초 보행**으로 설계한다. 같은 solver/반경/반복 변형이나 추가 RF 단독 모집 진단은 진행하지 않는다.

근거: `verification/banc-natural-gait-m3-20260919/parent-e286-original-final-review.json`, `parent-e286-final-source-review.json`, `parent-e286-output-manifest.json`, `e286-rf-state-attribution/`. 본 결과는 신체 보행 개선이나 제품 채택이 아니다.
