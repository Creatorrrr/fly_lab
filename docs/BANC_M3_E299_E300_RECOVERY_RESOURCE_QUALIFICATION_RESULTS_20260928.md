# E299/E300 — 실제 CUDA와 저장 상태 연결 검증

Root 설계 → 실제 gpt-6-sol 구현 → root 리뷰 절차로 새 회복 억제식 엔진과 whole-state wrapper를 구현했다. 엔진은 xhigh, wrapper와 E301 실험 연결은 high effort를 사용했다. 이 문서는 작은 회로의 수치·연결 검증이다. 실제 전체 BANC 보행 성과와 생물학적 기전 검증은 포함하지 않는다.

동역학은 `Rdot=(1-R)/tau-(u*r)*R`, 같은 RK 단계의 `g32=float32(min(1,R*(1+(tau*u)*float64(r))))`이다. source마다 한 번 계산한 계수를 기존 선택 연결에 적용한다. 계수1은 원래 packed CSR 식을 그대로 사용한다. 별도 CPG·위상·교사 궤적·fit을 추가하지 않았다. source/graph/근육 전달·원래 적분 순서와 모델 identity 경계를 보존한다. 이는 측정한 시냅스 생리 모델이 아닌 공학적 가설이다.

첫 리뷰에서는 CPU52 tests와 독립41cell/117edge/7tick 계산을 통과했다. 별도6cell/1tick 검사에서 작업 버퍼 교체 후 advance 거부 누락을 확인했다. CUDA FTZ의 subnormal sign/conversion 경계와 사전 설계의 CUDA2GiB cap 표현도 같은 Sol의 R1에서 보완했다. FIRST 소스와 실패 영수증은 보존했다. Root SECOND에서 수정 diff를 읽고 CPU59 tests/5 CUDA skip, Ruff, 독립 계산을 다시 통과했다. Wrapper는 root FIRST51 fake tests/독립 diskfuture 검사에서 승인했으며 수정이 필요하지 않았다.

RTX 4080의 실제 CUDA 검사5개가 통과했다. 독립41cell 공동 RK 계산과의 최대 절대 차이는 발화율0, adaptation `7.450580596923828e-9`, resource `4.779510121011299e-13`이다. 고정 한도는 발화율/adaptation `2e-5`, resource `2e-8`, 상대오차0이다. Neutral u0의 원래 계산과 비트 일치, 출력 마스크, 캡처/비캡처 경로와 diskfuture, 정상 control의 단일 host packet, sticky fault, 음의 최소 subnormal 거부, non-FTZ 형 변환을 확인했다.

E300 wrapper에 실제 새 CUDA 엔진을 연결한4cell 검사도 통과했다. 몸체·어댑터는 생성 fake scaffold였고 실제 물리는 실행하지 않았다. 비영 상태를 disk에 저장하고 새 소유 세션으로 복원한 뒤 동일 미래를 정확히 재현했다. 원래 whole-state 필드와 source identity를 검증했으며 u0 resource는1이었다.

전체 tiny 자격 supervisor10.452957300초, host tree peak1,042,329,600 bytes이다. 실제 tiny neural 생성14회(GPU12/CPU2), science GPU220 ticks/CPU107 ticks, setup·capture Python step180회, 직접 device kernel5회, 별도 conversion compile1회다. Torch tensor peak는 수치 검사58,368 bytes, wrapper 검사75,264 bytes이며 driver context는 이 수치에 포함되지 않는다. 모든 소유 PID 종료를 확인했다. 전체 BANC 원본/body/물리/solver/fit0이다. F2 전체모델/F1–F8 완료·후보 채택·F6 노출은 모두 false다.

원본 누적 `8078.0529733995245 s`를 이 검증 비용으로 초기화하거나 대체하지 않았다. E301 실제 신체 실험은 별도의 고정 입력·조건·원본 자원 예약을 사용한다.

근거: `verification/banc-natural-gait-m3-20260919/parent-e299-first-source-review.json`, `parent-e299-source-review.json`, `parent-e300-source-review.json`, `parent-e299-e300-tiny-qualification.json`, `parent-e299-e300-tiny-supervisor.json`.

승인 엔진 SHA `ef736e2a2d5afa0a021d9563e49ab4ec5600fc4156a6bf7b6f90ddb628d90fa0`, 엔진 tests `d7cf1aa466e8e4ed53658e227b554a40830d917194a3077c580103b12bbfa98f`, wrapper `05334501c9a6e07e5c4342765319e329ac08ba4c1f1711c66793e33c9b2f34b6`, wrapper tests `14ce5fd509bf7e7c3298002f8c9fef10b4dd34386ccbc42e4ebb37a129963e70`.
