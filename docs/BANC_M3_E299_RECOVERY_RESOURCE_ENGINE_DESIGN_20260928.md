# E299 — 같은 RK 단계의 회복 억제 신경 엔진

Root 설계, 2026-09-28. E298 저장 대수 검사는 controls1..100 활성1513 source 중1077에서 비unity, 같은-state 양수 입력손실이 기존R식의5.57241%로 남음을 확인했다. 이는 새로운 coupled 궤도의 증명이 아니다. 이 단계는 해당 식을 독립 엔진으로 구현하고 작은 생성 회로에서 정확성·상태·CUDA 실행을 자격 확인한다. 전체 BANC나 신체 실행/피팅은0이며 기존 E289/E290/E296 소스와 원본 결과는 동결한다.

## 고정 모델과 계산

신규 `tools/banc_m3_recovery_resource_synapses.py`의 `RecoveryResourceNetwork`는 원래 MotorTimescaleNetwork의 r/z와 공유 source R를 동일 RK4 단계에서 적분한다. 자원은 `Rdot=(1-R)/tau-(u*r)*R`, 계수는 float64 순서 `a=R*(1+(tau*u)*float64(r))`, `g32=float32(min(1,a))`다. Mask는 원래 outgoing current에만 적용하며 계수/Rdot는 같은 단계의 unmasked rate를 사용한다. R/u/tau float64, rate/mask/weights float32, FMA off, 원래 CSR packed warp/FTZ 합순서를 유지한다.

원래 selected edge 위치와 부호/weight bytes, source공유, explicit-zero CSR 저장, 모든 neuron/기존 readout/적응 상수는 그대로다. u0 또는 g32==1이면 원래 `weight*masked_rate` 식을 직접 사용한다. 비unity selected만 `weight*(masked_rate*g32)`를 원래 product/FTZ 순서로 계산한다. 계수는 고정 source당 한번씩 계산하는 device kernel과 재사용 buffer로 제공해 각 outgoing edge에서 float64식을 반복하지 않는다. 모든 stage에서 factor 계산→CSR current→자원 미분/nextR→원래 rate/adaptation 갱신 순으로 버퍼 의존성을 지킨다. 네 번째 단계의 R finish가 최종 R범위도 검사한다.

기존 resource dt guards를 유지하되 그 조건만으로 전체 회로의 안정성을 주장하지 않는다. uncapped r*g 국소 기울기 R*(1+2*tau*u*r)가1을 넘을 수 있다. 외부 clock/CPG/teacher/phase/trajectory/rootforce/원래CSR 추가삭제/seed격자/파라미터 fit은 없다. 수학 성질과 물리적 보행 또는 측정 생물학을 분리한다.

## 상태와 유효성

새 schema `flylab.m3-recovery-resource-synapses.v1`와 PRECISION/식/guard 계약을 포함하는 새로운 identity를 사용한다. Wrapper 호환용 metadata API `resource_metadata`, `checked_resource_metadata`, `resource_identity`, `checked_selection`을 제공한다. Snapshot key는 기존명 `recurrent_resource`/`synaptic_resource`를 사용할 수 있으나 새 schema/identity를 강제해서 기존 R 엔진의 checkpoint를 혼용하지 않는다. 원래 base tick0/rate0/z0/drive0/mask1만 import하며 이외 이전모델의 신경 상태 이식은 허용하지 않는다.

R은 유효 상태에서 모든 source에 0≤R≤1, u0에 R1이어야 한다. 실제 계수식이 적용되는 u>0 source의 unmasked stage rate는 r≥0/finite이어야 한다. 그외 rate는 기존 native−1e−4 허용 계약을 유지한다. E298는 저장값을 엄격히 검사했고 이 단계는 **내부 RK 단계**까지 검사한다. RK4 target clamp만으로 중간 rate 양수를 가정하지 않는다. 음수 rate/R 또는 범위 이탈을 clamp·보간·자동수정하지 않고 실패시킨다.

CUDA는 모든 RK factor 준비 단계와 최종 resource finish에서 해당 값을 검사하여 고정주소의 sticky device fault buffer에 기록한다. Invalid R/u0R/active rate/nonfinite factor는 허용되지 않는다. 최소한 fault 발생 종류를 구분할 수 있는 bit flag를 남긴다. Flag는 reset 또는 유효 restore에서만 해제한다. 새 buffer/parameter의 pointer와 bytes를 원래 방식으로 소유·검증한다. Factor는 stage별 유도량이며 checkpoint에서 별도 동적 상태로 저장하지 않아도 되지만 상태에 의존하는 hidden history를 추가하지 않는다.

Public advance/readout/snapshot은 fault나 최종 resource 오류를 검출한 뒤 성공값을 반환하지 않는다. CUDA `advance_readout`은 기존 전체-network rate 유효성 flag와 새 resource flag를 동일 readout packet에 합쳐 host 동기화를 추가하지 않는다. 원래 명시적 steps/tick/drive 검사 및 scientific/setup 구분을 보존한다. CUDA graph 재생도 동일한 고정 buffer를 사용한다. CPU도 같은 단계와 엄격 오류 계약을 갖는다. Fault 상태는 저장/후속전진으로 정상화할 수 없고 유효 reset/restore가 필요하다. 유효 snapshot에 faultbuffer는항상0이므로 checkpoint에 임의의 무효 상태를 정상으로 저장하지 않는다.

Reset/restore는 source parameter/메타데이터/shape/dtype/finite/원래 neural metadata·mask·R와active r를 쓰기 전에 검증한다. 기존 identity와의 혼용, source weights 또는 tensor pointer 변경, 누락/추가 필드는 거부한다. 비영 disk checkpoint→동일 미래가 exact여야 한다. Neutral u0 모든source에서는 원래 rate/adaptation/drive/mask/tick 결과를 bitexact(±0 포함) 유지하고 R은1이다.

## 구현·검증 범위

실제 gpt-6-sol/xhigh가 위 신규 엔진과 `tests/test_c_banc_recovery_resource_synapses.py`만 구현한다. 원래 모듈 전역 monkeypatch/편집은 금지한다. 승인 E289를 상속·읽기 재사용할 수 있으나 동역학이 바뀌는 CUDA source/CPU oracle/stage 순서/metadata/restore/guard는 새 식으로 일관되어야 한다. Public class name과 wrapper용 API를 구현 초기에 알린다. 생성 test만 실행하며 원본 전체 graph/실데이터 load/model/body를 생성하지 않는다.

Root FIRST→필요시 같은Sol R1→root SECOND→root 잔여수정. 생성 CPU 검증120초/1GiB/CPU1, n≤64/nnz≤256, scientific tiny ticks 총10000 이하. 모든 생성자/실제 step/setup/capture/advance/tick/실패 비용을 기록한다. 전체BANC/body/fit/solver0. 검사 항목은 독립 스칼라 및 packed 전류(혼합 u, signed/±0/subnormal/mask0), 상수 rate 해석 resource/unity, 작은 coupled CPU RK4, stagefault와 최종 resourcefault의 성공 readout 차단, 실제 parameter pointer, 엄격 snapshot/restore와 disk동일미래, source/formula identity 구별이다. 여러 grid/seed를 탐색하지 않는다.

Root 승인 소스 SHA를 동결한 뒤 별도 bounded tiny CUDA 자격120초/2GiB host/CPU1을 실시한다. CUDA측 n≤64/nnz≤256, GPU logical ticks≤2000, setup/capture 별도, GPU tensor 할당 peak≤512MiB(드라이버 context 제외)를 기록한다. 원본 GPU가 유휴인지 확인하며 실제 device/provenance를 저장한다. CUDA/CPU same-RK oracle 차이는 r,z absolute2e−5, R absolute2e−8/rtol0을 기존 자격과 동일하게 고정한다. Coupled stage가 동일 state를 쓰는지 root 독립 reference로 확인하고 frozen neutral bitexact, packed oracle, masked active-resource evolution, captured/noncaptured + disk동일미래와 faultflag를 확인한다. 임계값을 결과에 맞춰 변경하지 않는다.

Tiny PASS는 F2 전체모델 검증이나 F1–F8 완료가 아니다. E300 whole wrapper가 준비되어도 원본 body 실행은 별도 root 설계와 source-bound 자원 예약이 필요하다. 현재원본8078.0529733995245/8150초·예약0·모든실패·M2/E150/UI/F6미노출을보존한다.
