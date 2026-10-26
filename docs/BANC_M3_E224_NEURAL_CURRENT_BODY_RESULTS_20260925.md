# E224: 원래 신경 계산 보존과 실제 3초 몸체 비교

두 경로 모두 같은 E150 출발 상태에서 실제 전체 BANC/CUDA·FlyBody 3초 건강 기준을 통과했다. 별도 신경 전류 기반 effector는 재귀 MN rate 교체였던 E222의 0.2244초 건강 중단을 재현하지 않았다. 생성 검증에 이어 실제 몸체에서 사용할 수 있는 경로임을 확인했지만, 자연 보행이나 생물학적 MN 발화의 검증 완료는 아니다.

| 같은 0–3초 구간 | 원래 rate + canonical | 별도 effector + canonical | 기존 E150 저장 기준 |
|---|---:|---:|---:|
| 초기 전방 기준 전진 | 4.00356mm | 4.76240mm | 4.93456mm |
| net/path | .53671 | .54989 | .55345 |
| native 최소 upright | .92020 | .94094 | .92555 |
| 비발 접촉 | 0 | 0 | 0 |

새 경로는 canonical 대조보다 약18.95% 더 전진했고 최대 기울기가 줄었다. E150보다 더 멀리 전진한 것은 아니다. 3초는 최종 F3–F5의 10초 창을 대체하지 않는다. 원래 고정 평가기를 0.5–3초 prefix에 적용한 진단에서는 최장 전진 bout이 대조 .185초, effector .225초로 모두 .3초 미만이고, eligible cycle은 여섯 다리 모두0이다. 따라서 좌우 반주기 협응 및 반복성 개선은 아직 입증하지 못했다. effector 앞·뒷다리 knee 폭은 LF72.08/RF71.33/LH46.53/RH47.54도였다.

부모 독립 감사는 같은 시작 body/action/감각/clock, 391개 실제 neuronal readout과 별도 decoder input, 근육 EMA/관절/tendon 계산, 모든 raw 및 manifest를 확인했다. 관절 명령 재구성 최대 오차는 effector `4.663e-15rad`다. 첫5ms의 전체158706 neuron rate/adaptation/drive/mask 및 기존 신경 metadata는 effector patch만 제거했을 때 대조와 바이트 단위로 일치했다. 이후에는 몸체가 바뀐 감각 피드백이 들어오므로 신경 상태 동일성을 요구하거나 주장하지 않는다.

Sol/high 구현 → 부모 FIRST 수치 오류 기록 결함 → 같은 Sol 첫 수정 → 부모 SECOND21 관련 tests/Ruff → 부모가 남은 +Inf 최소값 처리1건 수정 →9 focused tests/Ruff 순서로 완료했다. 수치 실패의 원자료를 숨기지 않고 JSON 불가 값과 알 수 없는 시간은 명시적 null로 기록한다.

원본 process1, arm2, retry/fit/solver0. 과학 neural24000ticks/physical60000steps(각각6초), 실제 warmup520ticks/capture260ticks를 분리했다. supervisor `87.61998210000456s`, 내부 `83.82361110000056s`; 최대 개별 RSS `2,858,151,936B`, tree `2,819,018,752B`, 관측 추가GPU `471,859,200B`, 출력 `105,760,914B`. 소유 PID 모두 종료. 원본 누적 `5906.159917399684/6004s`, 잔여 `97.84008260031624s`, 증액0.

부모 raw 감사 `.8026022s`(process1.9438175s), prefix 분석1.9216434s(process2.2376333s)는 별도 비용이다. prefix metadata 경로의 첫 정적 준비 오류 `model_hash` 대신 실제 `modelHash`를 확인하여 바로잡았고, 오류1.3048223s/조회.4527557s 및 원본 실행0을 `parent-e224-prefix-analysis-preparation.json`에 보존했다.

증거는 `verification/banc-natural-gait-m3-20260919/e224-neural-current-body/`, `parent-e224-source-review.json`, `parent-e224-output-review.json`, `parent-e224-prefix-quality-summary.json`, `parent-e224-supervisor.json`이다. motor cut/실제 body clone 미래 비교는 수행하지 않았다. 전체 F1–F8, F2 기전·인과, F6/F7은 미완료이고 E150 best/M2/UI를 유지한다. 다음은 저장된 동일 후보3초 상태를 정확히 복원하여10초 고정 창까지 연장하는 별도 E225다.
