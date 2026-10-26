# E182: E181 자원 중단의 고정 후보 재검증

## 목적과 현재 근거

M3 finite-goal F1–F8은 그대로 유지한다. E181은 실제 시간응답 최적화 구현과 zero-theta 기준 재현을 확인했지만 LM plus가1.415초에서 자원 제한에 걸렸다. 과학적 후보 효과는 not-tested다. LM driver 시작0.732초 뒤 등록된 별도 이미지 작업이 현재도 살아 있으며 queue running1이다. 공유 자원 경합은 가능한 설명이고 아직 확정 원인은 아니다. 지난 turn은 구현·원자료 검증과 보존 확인 및 권위 있는 checkpoint 갱신을 마친 progress다.

E182는 같은 후보의 유효한 실제3초 값을 얻어 E181의 탐색을 이어갈지를 결정하는 기술 복구다. 새로운 기전이나 재조정 실험이 아니다. 기존 실패, 소스, 동결 설정, 원자료, 누적 사용량을 보존한다.

## 변경·고정 및 재사용 경계

| 구분 | 명시적 계약 |
| --- | --- |
| 변경 | 새 실행 경로와 자원 준비 확인·실행 중 관측·복구 비용 기록 |
| 고정 | 전체158706 BANC/CUDA, FlyBody, 신경·물리 동역학/입력/평가기, neutral 초기 상태, 기존5/10/40ms 시간응답 및 양수 low/high 합 보존 tilt |
| 순서 | baseline theta=(0,0,0,0,0,0), LM plus=(0,.125,0,0,0,0), LM minus=(0,-.0625,0,0,0,0) |
| 과학적 길이 | 각각600control=3초, control당20neural/50physics. 원래 startup99 보호와[.5,3)평가 창 |
| 판정 | 기존 전진 속도 부족 J, 초기 보호, 건강성, 원래 baseline 전진/net-path95% 이상·slip 비증가, 최소1% J개선 및 기존 tie-break |
| 후속 | LM 두 팔이 유효하고 부모 원자료 리뷰를 마친 뒤에만 나머지4축을 별도 구체화. 본 계약은3팔만 |

소유 신규 파일은 tools/recover_c_banc_m3_native_timing.py와 tests/test_c_banc_native_timing_recovery.py 두 개다. E181 소스는 수정하지 않는다. 불필요한 모델 루프 복제를 피하기 위해 기존 E181의 verified_inputs, run_arm, supervise_command 및 순수 비교/선택 함수를 재사용한다.

**이중 검증의 의미:** 실제 보존된 E181 freeze·source review·baseline review·LF 결과를 원래 verified_inputs로 검증한 반환값만 E181 run_arm에 전달한다. 기존 gate의 값/스키마/해시를 조작하거나 새 gate로 가장하지 않는다. 이 값은 원래 과학적 구현의 동일성을 확인하는 재사용 함수의 전제일 뿐이며, 새 실행의 허용 범위는 별도 E182 freeze와 부모 source review가 결정한다. E182가 별도로 정확한3개 theta·zero 초기화·새 output·공유 자원·누적 한도·실행 순서를 검증한 뒤만 원본을 읽거나 모델을 구성한다. E181 arm 원자료의 E181 스키마와 원래 과학 계약 해시는 그대로 두고, 새 E182 provenance가 실제 E182 freeze/source/arm/기존 gate와 결과 해시를 연결한다. E181 driver나 실패 campaign은 재개하지 않는다.

E182 baseline이 기존 E181 baseline/E180 reference와 raw prefix가 일치함을 부모가 확인하고 새 baseline artifact 해시에 묶인 E182 baseline review를 작성해야 두 후보를 시작할 수 있다. plus의 완전한 결과와 startup·상태 검산을 부모가 확인한 별도 review 뒤 minus를 시작한다. 부분 결과, 기술 오류, review 부재는 다음 팔을 허용하지 않는다. 과학적 건강성/초기 보호 실패도 구분해서 보존하며 해당 복구 비교는 종료한다.

## 공유 자원 관측

실행 전 HTTP GET http://127.0.0.1:8188/queue의 running/pending 개수와 기존 queue client PID16248/35496의 생성시각을 함께 검사한다. PID 재사용을 구분한다. 알려진 작업뿐 아니라 queue의 새 작업도 검출한다. 원래 client 생성시각은2026-09-23T14:37:08.417602+00:00와14:37:08.433284+00:00다. 두 관측을2초 간격으로 시행하며 모두 empty/not-live일 때만 시작한다. 연결 실패, 알 수 없는 상태는 준비 확인 미완료로 기록하고 native를 시작하지 않는다. 단일 GPU 이용률 값으로 준비 완료를 단정하지 않는다. CPU 설계/합성 검사에는 이 조건을 적용하지 않는다.

driver에서 native child를 감독하는 동안2초 간격으로 queue 개수와 nvidia-smi의 timestamp/device/utilization/memory/power/clocks를 JSONL로 기록한다. 다른 작업의 prompt·파일·이미지는 읽지 않고 process를 종료하거나 설정을 바꾸지 않는다. subprocess/HTTP 각각2초 이내 제한, 오류를 기록한다. 관측기는 child의 모델 상태·입력·물리 주기를 바꾸지 않으며 finally에서 정리한다. GUI 창을 만들지 않는다. raw stream과 native child PID/생성시각 및 소유 후손 종료는 기존 E181 감독을 재사용한다.

시작 후 다른 작업이 나타나면 현재 팔은 원래 시간 한도 안에서 정리하고 다음 팔의 시작은 준비 확인까지 보류한다. 완료된 원자료의 보행 지표와 자원 독점 여부는 별도 판정한다. 동시 작업이 있었다는 이유만으로 실제 완료한 과학적 값을 지우거나 보행 실패로 바꾸지 않는다. 계측 미확인/중첩 상태에서 속도 개선·경합 원인의 인과 주장은 하지 않는다.

## 자원 및 실패 규칙

기존 보수 누적3697.5422548998613/4994초를 승계한다. 최대3 native child, 각internal100/external105/supervisor110초, 자동 재시도0이다. startup/사전 원본 검산10초를 포함해 최대340초를 기존 남은1296.4577451001387초 안에서 예약한다. CPU1/GPU1, 한 번에 한 소유 native child만 실행한다. 각 driver 전체150초 이내, 외부 감독165초로 자원 준비/해시/계측 정리도 제한한다. 원래 process당300초를 넘기지 않는다. 기존 free20GiB/arm1GiB, 복구 전체4GiB 한도를 적용한다. 새 overall 예산 증액·후보 수 확대는 없다.

시작하지 않은 resource-blocked 호출은 native0으로 기록하되 관측/검증 CPU 시간은 별도 보존한다. native가 시작된 실패는 실제 elapsed와110초 중 보수 값을 기록한다. 소유 PID·birthtime과 ledger가 없는 비용/진행을0으로 만들지 않는다. setup 실행/capture/unknown, 실제 neural/physics/complete control을 구분한다. 첫 기술/자원 오류에서 복구 native sequence 종료, 결과와 raw를 보존한다. 명시적으로 완료된3초 팔만 objective_completed에 센다. 원래 E181 partial도 누적 attempts에서 제외하지 않는다.

## 구현 및 검증 순서

부모 설계 → actual gpt-6-sol/high 첫 구현 → 부모 FIRST → 필요한 첫 수정은 같은 Sol → 부모 SECOND와 필요한 잔여 수정. 부모가 frozen metadata와 source review를 작성한다. Sol은 지정된 두 파일만 쓰고 원본 배열·모델·GPU·fit을 실행하지 않는다. 합성 검사에서는 busy/unknown/reused PID·readiness 전 native 금지·잘못된 theta/source/order 거부·child 실패/부분값 비용·관측기 종료를 중심으로 확인한다. 숫자 계약을 복제하는 장식적 테스트는 추가하지 않는다.

실제 baseline의601control/30000native 및 원래 full state를 기존 raw와 검산한다. 후보에서는 시작 상태 동일성, 고정 파라미터, 실제 tilt, bank→activation→joint 명령과 시계, startup 및 J/physical summary를 부모가 검산한다. 테스트/소스 검증은 최종 F3–F5나 자연 보행의 증거가 아니다. F6 미노출·M2/UI·기존 실패를 유지한다. 결과는 기존 verification/banc-natural-gait-m3-20260919/e182-native-timing-recovery와 계획/진행 기록에 보존한다. 전체 목표는 active다.
