# E193: 여섯 다리 감각 전달의 동일 궤적 관측

2026-09-24, 원본 실행 전 설계. 기존 finite-goal F1–F8, M2/UI, 미사용 F6, 실패 기록과 원본 누적 4670.873046799719 / 4994초(잔여 323.126953200281초)를 유지한다. 이전 턴은 E192의 실제 적합과 독립 검산으로 부적격 전달 가족을 좁혔으므로 progress이며, 보행 개선은 아니다.

## 목적과 구별할 설명

전체 감각 경로는 연결되어 있다. 현재 남은 질문은 물리적 무릎·접촉 변화가 여섯 다리의 수용기 특징, 지연된 외부 입력, 실제 신경 상태 중 어디에서 구별력을 잃는지다. A는 유의하게 변하는 물리량이 특징의 영점/포화 또는 신경망의 음수 순입력/포화 구간에서 평탄해지는 경우, B는 각 단계에 변화가 남아 단순 감각 증폭·지연 제거를 선택할 근거가 없는 경우다. 한 궤적의 연관성은 인과 증명이나 생물학적 튜닝 증명이 아니다. 수치 조정이나 후보 채택은 이번 관측에 포함하지 않는다.

읽기 감사에서 E150 연속 자료에는 감각 포트 벡터가 없고, E177/E188은 RM 일부 세포 또는 MN 하나의 입력만 관측했다. 저장 contact에는 인코더가 쓰는 부착력 차감 support/non-floor touch가 없으며 기하적 무릎 속도는 live Jacobian을 사용한다. 체크포인트만으로 중간 필터 상태를 정확히 복원하지 않는다. E65/E66 지연 제거, E39/E40 hair plate, E51 고정 신체 극성 교란, E176/E177 RM 감각 연결 복원의 실패를 보존한다.

## 실행과 불변 조건

부모 설계 → 실제 gpt-6-sol/high 최초 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 필요한 잔여 수정은 부모가 수행한다. high는 원래 실행을 바꾸지 않는 관측 및 지연/종료 경계 정렬의 난이도 때문이다. Sol 소유 파일은 `tools/observe_c_banc_m3_sensory_transmission.py`와 `tests/test_c_banc_sensory_transmission.py` 두 개다. 기존 실행기/물리/신경/평가기/자료를 수정하지 않는다.

E181의 실제 baseline `verified_inputs`를 통과하고 같은 E150 candidate-neutral, theta=(0,)*6, 동일 graph/network/adapter/body에서 `run_arm` 한 번만 실행한다. 기존 E188의 정확한 E185 baseline 비교 및 resource-readiness 보조 함수를 재사용할 수 있다. 소스/설계/입력 SHA와 부모 승인 검토를 원본 생성 전에 요구한다. 큰 입력 해시는 부모가 한 번 검증하고, 살아 있는 소유 부모 PID/생성시간, 정확한 SHA 목록과 file stat, 신선한 readiness를 가진 영수증만 자식에서 사용한다. 검증 결과를 위조하거나 E181 게이트를 건너뛰지 않는다.

600 control / 12000 neural / 30000 physics / 3초를 관측한다. 시작과 600개 종료 경계 601행을 E185 baseline과 비교한다. trace 샘플 전체, native recruitment/contact 전체, 시작/최종 snapshot의 실제 배열 및 모든 상태 값이 같아야 한다(기존 비교의 wall_s 예외만 유지). 한 항이라도 다르면 원본의 물리적 질문은 invalid이다. 새 감각·근육 계수, 신경 연결/효능, 감각 지연, 초기상태, 외부 CPG/교사, 재실행, F6 노출은 모두 0이다.

## 관측 위치와 원자료

1. E150 원래 관측 callback의 성공 뒤 시작 anchor에서 가장 안쪽 RateBodyAdapter를 찾아, 실제 객체의 encode를 관측 wrapper로 감싼다. original encode의 인수·반환 객체·예외를 보존한다. encode 안에서 호출하는 body.leg_observation 및 knee_kinematics를 임시 wrapper로 감싸 실제 반환된 관측을 복사한다. 원래 함수는 각각 정확히 한 번 호출하며 추가 물리 계산을 하지 않는다. 성공/실패 모두 기존 인스턴스 속성의 유무까지 finally로 복원한다. 같은 body 함수를 쓰는 다른 단계는 감싸지 않는다.
2. encode 호출 tick 0..599의 원래 관절 각/속도 42, support/touch 6, geometric knee 각/속도 6, old/new filtered 및 old/new velocity_lowpass, 실제 last_features와 반환 sensory drive를 저장한다. q/v의 knee 대체 및 request는 고정 기존 식으로 별도 재구성한다. gain=18/load multiplier=1/front hair plate=false를 실제 객체에서 검증하고 저장한다. gain 단위를 측정 mV/Hz로 부르지 않는다.
3. runtime의 실제 포트 순서(leg/kind/cell_type, graph index/root ID), unique sensory index와 port expansion 순서, joints/coxa sign/neutral, spec 및 네트워크 cap/coefficient/threshold를 보존한다. 순서는 추측하거나 알파벳순으로 바꾸지 않는다. unknown claw/hook의 0 특징도 제외하지 않는다.
4. 경계 tick 0..600에서 전체 감각 세포의 float32 rate, 외부 drive, adaptation, output_mask 및 기존 CUDA `_current(network.rate)` 결과를 저장한다. 기존 신경 상태와 시계를 읽기 전후 검사한다. 별도 neural step은 없으며 intermediate RK 전류를 관측했다고 주장하지 않는다. graph.n=158706 및 실제 packed CUDA/device/identity를 검증한다. MN/96근육·신체 출력은 동일 baseline raw에 이미 있으므로 중복 모델을 만들지 않는다.
5. encode tick k의 반환 감각 drive는 종료 경계 k+1의 network.drive와 bit-exact이어야 한다(하행/항법 입력과 sensory index의 겹침이 없음을 실제 검사). 반환 drive는 old filtered를 float32에 기존 순서로 더한 값과 exact, 새 필터 및 LP는 같은 식과 atol 1e-12/rtol 0, 특징은 기존 typed 식과 atol 1e-12/rtol 0이다. LP/filtered 연속성 및 각 encode의 before clock == k*20/k*.005, after clock 동일, 종료 관측 시계도 검사한다.
6. `encoder-signals.npz`, `sensory-signals.npz`, `runtime-provenance.json`, baseline raw, `validity.json`, `analysis.json`, `terminal.json`을 새 출력 디렉터리에 남긴다. 실패 시 수집한 행과 availability/complete/error를 보존하고 원래 예외를 cleanup 예외로 덮지 않는다. 기존 출력 충돌은 실행 전에 거부한다.

## 고정 분석과 판단

startup encode tick 0..99 및 evaluation 100..599를 별도로 모두 보고한다. 포트마다 실제 knee/support/touch, feature, request, old filtered drive의 min/max/p5/p95/std와 정확한 feature==0/==1 비율, 연결된 세포들의 rate p5/p95 및 cap의 95% 이상 비율, endpoint net=drive+current-threshold-adaptation의 비양수 비율을 보존한다. 세포별 원자료와 포트별 표를 모두 유지하고 임의 제외나 lag/gain 탐색은 하지 않는다. 조건이 불일치하는 단위/단계는 별도 표로 구분하며 하나의 비율로 합치지 않는다.

예정된 해석은 (a) 물리량 변화와 특징 범위를 함께 보아 인코더의 제한 여부, (b) 동적 특징·drive가 존재하는데 endpoint rate가 평탄한 경우 현재 신경 상태가 입력을 억제/포화시키는지, (c) 다리별 신호가 유지되면 단순 감각 누락 설명을 배제하는 것이다. 수용기 공식 자체가 생물학적으로 옳다는 판정이나 개선 계수 선택은 없다. endpoint net과 rate는 시상수 때문에 즉시 같을 필요가 없으므로 rate==cap 또는 rate==0을 net의 대체로 쓰지 않는다. 개선 근거가 불충분하면 그대로 inconclusive로 보고하고 다음 후보를 자동 만들지 않는다.

## 자원과 검증

원본 단일 process/3초 native 1회, GPU 1, CPU/BLAS 1, retry 0. 기존 run_arm 내부100초를 유지하며 새 도구 내부140초, supervisor 전체150초(실행145+소유 자식 cleanup5), 출력2GiB/그 안 baseline1GiB, free20GiB를 적용한다. readiness 두 번 및 실행 중 자원 확인은 기존 resource_readiness를 사용한다. 실제 감독 wall을 기존 누적에 더하고 준비·합성 시험·부모 검산은 별도 기록한다. 미집계 비용은 0으로 적지 않는다.

시험은 wrapper 실제 호출 수/반환·예외/복원, 지연된 drive 정렬, 중복 sensory index의 float32 가산, 필터/velocity LP 및 typed polarity, 부분 실패 보존, stale gate/출력 충돌을 다룬다. 부모는 원자료에서 별도 식으로 특징·필터·drive·시계·통계와 baseline 동일성을 검산한다. 원자료 유효성 및 관측기 시험 통과는 자연 보행 개선이나 목표 완료가 아니다. 본 설계 저장 시점 원본 E193=0, 후보 미채택, 전체 active·미완료다.
