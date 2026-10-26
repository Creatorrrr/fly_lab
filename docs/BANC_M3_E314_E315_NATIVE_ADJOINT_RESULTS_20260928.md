# E314–E315: 현재 BANC의 CUDA 학습 계산 검증

현재 모델의 신경 연결 효능을 학습하기 위한 sparse current와 SSP 역미분 계산을 구현했다. 실제 전체 BANC에서 기존 출력 재현과 **1ms 구간·한 방향의 국소 미분**을 검증했다. 학습 실행·새 보행 후보·제품 채택은 아직 0이며, 자연 보행 finite-goal/F1–F8은 계속 미완료다.

## 출발점의 해석 정정

E150의 정식 적격 보행 주기가 0이라는 결과는 신경 리듬 자체가 없다는 뜻이 아니다. 기존 E152에는 전체 창에서 각 다리 21–22회 반복이 기록돼 있고, 전진이 46회 끊기며 최장 전진 구간이 .215초여서 적격 주기가 없다. E163의 source 위상 결집도 높았다. E73의 한 개발 조건 F4/F5 부분 통과는 원형 경로에 따른 F3 실패와 분리한다. 다음 학습은 추가 리듬 생성보다 **한 주기 안의 근육 동작과 추진 유지**를 목표로 해야 한다.

E313의 수신 입력 감쇠 후보는 실제 .2156초 upright 기준 미달로 기각된 상태를 유지한다. 같은 λ 탐색, 고정 teacher 절대 위상/12Hz 맞추기 또는 이미 실패한 단일 RM 전달 보정을 반복하지 않는다. E24/E25의 큰 장기 형식 미분과 실제 float32 변화의 불일치도 이번 짧은 국소 통과로 해소됐다고 주장하지 않는다.

## 구현·리뷰

- `tools/banc_m3_grouped_current_adjoint.py`: 원래 packed CSR lookup을 유지한 기존 간선의 양수 그룹 효능과 희소 surrogate VJP. 변경 효능 때문에 새 가중치 종류가 늘어도 원래 14-bit lookup을 다시 만드는 방식에 의존하지 않는다.
- `tools/banc_m3_ssp_adjoint.py`: 현재 per-cell physiology를 사용하는 functional SSPRK33와 세 stage 전체의 역미분. 외부 위상·CPG·교사·몸체 보정을 추가하지 않는다.
- `tools/verify_c_banc_m3_native_adjoint.py`: 동결된 실제 입력에서 중립 재현과 작은 효능 변경을 검사하고, 전체 endpoint 및 장치 완료가 확인된 tick을 기록한다.

Root 설계 → actual gpt-6-sol/xhigh 구현 → root FIRST → 같은 Sol의 첫 수정 → root SECOND를 수행했다. FIRST에서 효능 cache 변조 누락, CUDA 장치 index 비교, GPU 단계별 CPU 동기화를 수정했다. runner에는 endpoint 원자료와 GPU 실행 확인 회계를 보강했다.

SECOND 뒤 root가 CUDA의 매우 작은 값 경계를 추가 검산했다. CuPy의 암묵적인 float32→float64 변환이 subnormal cotangent를 지워, 기대한 rate VJP `1.7936620613e-13`과 그룹 VJP `3.2172235418e-56`을 0으로 만들었다. root가 명시적인 no-FTZ 변환으로 잔여 문제를 고쳤다. 기존 전방 CSR의 FTZ 규칙은 유지했다. 실패와 수정 전 소스는 별도로 보존했다.

생성 검증은 root CPU 52검사, 수정 후 CUDA 4검사, 독립 dense float64/유한차분, runner 15검사 및 실제 엔진을 사용한 39-cell CPU 통합을 통과했다. 독립 생성 미분의 최대 상대 L2 오차는 약 `1.85e-7`이었다. 초기 개발의 fixture·Ruff 오류와 subnormal 실패를 없었던 것으로 처리하지 않는다.

## 실제 전체 BANC 결과

원래 158,706개 신경과 11,584,852개 간선을 사용했다. 선택은 기존 premotor 7,136개 내부의 725,849개 간선·8,547개 공유 그룹이며, 현재 E150 CSR 값과 해부학적 선택을 그대로 유지했다. 원래 E29의 옛 가중치는 사용하지 않았다.

| 확인 | 실제 결과 |
|---|---:|
| 중립 효능 처음 20tick | 저장된 native SSP 전체 rate/adaptation과 byte exact |
| 국소 시작 | 기존 .5초 상태, 입력 고정 |
| 미분 구간 | 4tick = 1ms |
| 검사 손실 | 최종 premotor rate의 float64 평균 |
| 효능 변화 | `1 ± .001 × sign(gradient)`, 실제 float32로 반올림 |
| 관측 손실 차 | .004333240837976149 |
| 예측 손실 차 | .004332607082645144 |
| 상대 오차 | .0001462757455 = **약 .0146%** |
| 사전 기준 | 유효 변화 >1e−7, 같은 부호, 상대 오차 ≤5% |

E315는 valid/supported다. 이 결과는 해당 상태·구간·방향의 검증이며, 장기 미분·물리 되먹임·보행 개선을 증명하지 않는다. 검사 손실을 보행 학습의 목적함수로 사용하지도 않았다. root는 저장된 전체 신경 endpoint에서 평균과 예측 내적을 독립 재계산하고, source/input/output SHA 및 회계까지 **65검사**를 통과했다.

## 비용과 현재 상태

- E315 원본 1회, 재시도 0. 실제 확인된 forward 32tick(.008 신경 모델초), backward 4tick. 새 신체·물리·solver·fit 0.
- 원본 감독 wall **5.960796700033825초**, 신경 검사 및 준비 science wall 3.75947초. 추가 GPU peak reserved **352MiB**, process peak 약1.69GiB로 사전 한도 이내다. 이 수치를 기존 구현 대비 속도 향상으로 해석하지 않는다.
- 원본 누적 **8501.121034199608 / 8560초**, 잔여 **58.8789658003916초**, 예약 0, 소유 프로세스 모두 종료. 기존 비용을 유지했고 배정 증액은 없었다.
- E314 root 생성 CUDA는 실패·수정·재검사를 포함해 **19.2903777/60초**. root CPU 감독6.42934초, E315 생성 통합 감독7.91409초, 원자료 감사 .18659초(프로세스 .64394초)는 원본 실험과 구분한다. Sol의 생성 검사 시간과 일부 계측되지 않은 읽기/style overhead도 별도 기록에 남겼다.

다음 작업은 이 검증된 계산에 **현재 단조 근육 모집의 미분과 시작 조건 제약**을 연결하고, 절대 위상을 강제하지 않는 한 주기 내 운동 형태의 목적함수로 제한된 학습을 설계하는 것이다. 먼저 현재 decoder의 기록된 MN→근육→관절 출력 재현을 확인하고, 학습 구간의 유한 변화와 실제 startup을 함께 판정해야 한다. 조건부 학습을 통과한 동일 후보만 원래 실제 신체 startup .5초→짧은1.5초 판별→10초/F1–F8로 진행한다. E315의 신경 평균 상승을 후보로 저장하거나 현재 페이지에 적용하지 않는다. M2·활성 UI·미사용 F6·모든 기각 기록을 보존한다.

근거: `verification/banc-natural-gait-m3-20260919/parent-e314-generated-qualification.json`, `parent-e315-source-review.json`, `e315-native-adjoint-probe/`, `parent-e315-original-supervisor.json`, `parent-e315-original-raw-audit.json`. 사전 설계는 [E314](BANC_M3_E314_NATIVE_ADJOINT_DESIGN_20260928.md), [E315](BANC_M3_E315_NATIVE_ADJOINT_PROBE_DESIGN_20260928.md)다.
