# E258 저장6간선 후보 초기 오차 분해 결과 — 2026-09-25

E257 실패는 이전25간선 후보와 같은 muscle41 경로라는 H258을 지지하지 않았다. 원자료·판정 독립 검산1,106개는 통과했고 저장 자료 분석은 유효하다. 이 결과는 E257 기각을 유지하며 자연 보행이나 신경 인과성을 입증하지 않는다.

startup controls1..99의 새 임계 위반26개는 모두 LH이고, 양의 SSE 증가의99.5054%가 LH에 속한다. 하지만 최초/최대 위반 관절은 둘 다 column19, 실제 감각 어댑터 매핑의 `tibia_pitch`다.

| 고정 endpoint | control / 시간 | candidate 오차 | muscle44 tibia_extensor signed 명령 변화 | muscle43 accessory_tibia_flexor signed 변화 | muscle41 양의 오차 방향 기여 비중 |
|---|---:|---:|---:|---:|---:|
| 최초 새 위반 | 56 / .280s | −.040839938998rad | −.001388724446rad | +.000006583436rad | 0% |
| 최대 초기 오차 | 70 / .350s | −.046753214497rad | −.007582455378rad | +.000279629970rad | 0% |

두 endpoint에서 candidate error 방향으로 투영한 양수 기여의100%는 LH muscle44 `tibia_extensor_muscle`이고, muscle43은 일부 반대 방향으로 상쇄한다. muscle44의 고정 MN columns는[68,107]이다. 이 수치는 저장된 명령의 산술 분해이며 어느 신경 연결이 원인인지 아직 검증하지 않았다.

rest+.02×moment×activation으로 두 팔의 controls1..300 명령을1e−12rad 이내로 재구성했다. 전체301×42×96 signed 기여, 근육명/다리/MN 소속, signed·positive·negative SSE와 선형/제곱항을 보존했다. row0은 decode 전 잔차를 별도로 기록했다.

| original24 rate 차이 창 | RMS | 최대 절대 차이 | RMS/200 |
|---|---:|---:|---:|
| 초기200ms / 1..40 | .000001891638 | .000030517578 | .000000009458 |
| startup / 1..99 | .000135794571 | .0010986328125 | .000000678973 |
| steady /100..299 | .124826532628 | .933731079102 | .000624132663 |
| 전체 /0..300 | .103737689674 | .961585998535 | .000518688448 |

startup의 전체391 MN 차이는 RMS .027736554786, 최대 .745491027832다. original24의 첫 정확 차이는control8, 절대1e−5 초과는control35다. 감각 old_filter 관측t와 returned drive가 적용되는 endpoint t+1을 구분했다. 5ms 기록 간격으로 세부 신경 인과 latency를 주장하지 않으며, E256 tick40000 정적 입력 Q6와 폐루프 tick0 분석의 비율로 보호 성공을 판정하지 않는다.

실제 gpt-6-sol/medium 구현을 부모가 FIRST 검토했다. 기존 E256 저장 f64 가중치의 정확한f32 왕복 형식 허용, 신경 배열의 빼기 전f64 승격, 전체96 근육명·signed 기여 매핑의 세 결함을 같은 Sol이 첫수정했다. SECOND에서 생성10pytest/Ruff, 부모 독립124생성검사/실제CLI5검사가 통과했고 부모 제품 잔여수정은0이다. freeze-v1은 최초 freeze의 근육 표기만 실제 provenance 이름 `femur_reductor_muscle`로 실행 전에 정정했으며 구판과 정정 기록을 보존했다.

원본1/retry0, 신경·신체·CUDA·physics·fit·solver0, CPU1. 원본 감독 .750858899992초(자식 .501685799987초), 누적6996.279667499624/7074초, 잔여77.720332500376초다. host tree peak138465280B, 출력1620827B, 소유PID14868/18348/39024는 종료됐다. 부모 독립 검산 .726372199992초는 별도다.

다음 검토 대상은 고정 muscle44의 두 MN과 현재 근육 변환의 출력 민감도다. 기존 muscle41 가정은 기각한다. 이 결과를 근거로 gain sweep, 새 신체 실행,10초 채택시험,F6 노출을 자동 수행하지 않는다. F1–F8미완료/E150 best partial/M2/UI/이전 실패와 비용은 유지한다.

근거: `verification/banc-natural-gait-m3-20260919/e258-lh-partition-startup/`, `parent-e258-source-review.json`, `parent-e258-output-review.json`, `parent-e258-supervisor.json`.
