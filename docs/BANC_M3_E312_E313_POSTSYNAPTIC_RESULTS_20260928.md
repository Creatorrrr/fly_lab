# E312–E313: 입력 부하 조절 구현과 실제 신체 검증

**단일 lambda1 후보는 기각한다.** 실제 신체가0.21559999999997626초에 upright0.8996158047567534로 원래0.9 기준을 벗어났다. 초기0.5초 평가를 마치지 못했고,1.5초/10초 자연 보행 검증에 진입하지 않았다. Lambda0는 E307 SSP/u0의 첫 신경·신체 구간을 정확히 재현했다. 이는 구현 연결과 수치 검증의 성공, 보행 후보의 실패다.

## 구현과 리뷰

E312는 원래 P7136→P7136 간선725849개의 입력 절댓값 총량을 수신 뉴런별로 누적하고, 기존 membrane tau를 사용하는 h 상태로 전달 강도 g=1/(1+lambda*h)를 조절한다. 원래 graph158706/11584852·부호·모터·근육·신체·감각을 유지하고 source resource는 모두u0/R1이다. 이는 정규화한 공학적 부하 모델이며, 생물학적 수용체·막전위 복원의 검증을 뜻하지 않는다.

E313은 이 신경 상태를 whole session의 신체·감각·근육 bank·시계와 함께 저장·복원하는 wrapper와 단회 신체 검증 도구다. 원래 건강성·startup·short·F1–F8 기준을 유지한다.

사용자 요청대로 root가 설계하고 실제 gpt-6-sol/xhigh가 구현했다. Root FIRST에서 E312의 기존 진단 함수3개가 lambda1의 h를 누락하는 오류를 재현했다. 같은 Sol R1은 해당 모호한 진단 경로를 거절하고 명시적 joint-state oracle로 안내하도록 수정했다. E313 FIRST의 과거 E310 실행본과 현재 R1 소스 hash 충돌도 같은 Sol이 수정했다. Root SECOND 뒤 발견한 binary NPZ를 JSON receipt로 읽는 잔여 문제는 root가 수정하고 실제 NPZ 회귀 검사를 통과했다.

## 생성 검증

- Root CPU: 엔진45개, whole/runner54개, 추가 binary NPZ 회귀1개 통과. 실제4cell 신경과 fake whole의 nonzero disk future 정확성도 통과했다.
- 최초 CUDA pytest 묶음은 자식25초 상한에서 종료됐다. 부분 결과와 미계측 tick을 보존했으며 전체 PASS로 계산하지 않았다. 미실행 검사에 대한 별도 사전 계약 후 root 독립39cell 및 CUDA subnormal/pointer 검사를 완료했다.
- RTX4080 독립 CUDA 검산:117 logical ticks와37 scalar oracle ticks; 최악 발화율 오차9.5367431640625e-7, adaptation0, h2.6643028894213217e-9. 기존 허용치 이내였다. Lambda0 SSP bytes exact, lambda1 효과, graph/direct/disk future exact와 잘못된 상태의 거절을 확인했다. 알려진 GPU reserved peak8MiB.
- 최초 중단을 포함한 생성 CUDA 감독 비용35.88495789997978초. 과거60초 상한을80초로 확장한 사전 추가 계약을 보존한다. 전체 BANC 원본/실제 신체 실행 비용과 별도다.
- 실제 입력 gate에서 root 작성 절대 경로 metadata 오류와 NPZ reader 오류가 순서대로 발견됐다. 실패한 freeze V1/V2를 보존하고 새 V3에서 소스130개·입력2838개 pin 및 원래 전체 상태를 검증했다. 이 사전 단계의 모델/신체/신경 진전은0이었다.

## 실제 신체 결과

| 항목 | Lambda0 비교 | Lambda1 후보 |
|---|---:|---:|
| 완료 control |1|43|
| 실제 신경 tick |20|880|
| 실제 native physics step |50|2156|
| 신체 시간 |0.005초|0.2156초|
| 최소 upright |0.9976102815715782|0.8996158047567534|
| 비발접촉 |0|0|
| 결과 |기존 SSP/u0 첫 구간 exact|control44 도중 원래 건강 기준으로 중단|

후보의 신경은 control44용20tick을 먼저 완료했다. 신체는 그 구간의6번째100us step에서 중단됐으므로 마지막 완료 control의 진단은 neural tick860, partial-final 신경은880이다. 이 차이를 숨기거나 같은 시점으로 비교하지 않았다. 건강한 whole final은 생성할 수 없었고, 사용 가능한 partial network/adapter와 raw native 기록을 보존했다. F2 whole future 통과로 주장하지 않는다.

두 신체를 포함한 과학 실행은 신경900tick/native2206step, 별도 실행 setup560tick/capture 기록280tick/settling4000step이다. 원본1회/재시도0, 감독78.26429990003817초, 최대 tree RSS3388145664B, 출력614179028B. 모든 소유 프로세스 종료를 확인했다. 누적은 **8495.160237499575/8560초**, 잔여64.83976250042542초이며 예약은 해제했다.

## 실패 양상과 다음 판단

저장된 같은 시점의 E307 neutral과 후보의 prefix만 대조했다.0.1초에 수신 대상94.4%가 g<0.6,0.215초에는95.1%였다. 마지막 g 중앙값0.51345로, 대부분의 선택 재귀 입력이 빠르게 절반 가까이 감쇠됐다. 기존24core도 마지막 g0.5590–0.5754였으며, 같은 시점의 core 발화율 차이 RMS10.52가 관찰됐다. 전체391MN의 차이 RMS12.85, 관절 명령 차이 최대1.60498rad였다. 이는 관찰된 연쇄이며, 특정 회로가 자세 실패의 단독 원인이라는 인과 결론은 아니다.

이번 기전이 원하는 반복 보행을 만드는 증거는 없으며, lambda를 줄이는 grid나 실패한 source/decoder fit 반복으로 이어가지 않는다. 다음 설계 전 기존 E29 내부 재귀 학습의 실제 자유도·조건부 입력·적분·미분 검증과 실패 범위를 먼저 확인한다. 학습 가능성 전체를 이전1gradient/3step 실패로 부정하지 않되, 같은 실험을 이름만 바꿔 반복하지 않는다. 기존 최종 기준·M2·UI·F6 미노출을 유지하고 목표는 active/미완료다.

주요 근거: `parent-e312-tiny-qualification-v2.json`, `parent-e313-tiny-integration-v2.json`, `e313-postsynaptic-load-body-freeze-v3.json`, `parent-e313-original-supervisor.json`, `parent-e313-original-raw-audit-v2.json`, `parent-e313-saved-prefix-contrast.json`(모두 `verification/banc-natural-gait-m3-20260919/`). 첫 raw audit은 건강 중단에도 정상 final이 있다고 가정해 실패했다. V2에서 실제 partial-final 및 어긋난 신경/신체 시점을 명시적으로 검사하여 보완했고, 원본을 다시 실행하지 않았다.
