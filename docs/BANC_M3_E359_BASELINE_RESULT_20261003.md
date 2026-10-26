# E359 기준 모델 결과

기준 모델의 actual99-control 구간은 완료됐고 생산자 초기 조건은 PASS였다. 다만 실제 import된15개 프로젝트 소스가 사전 목록에서 빠져 **전체 실행은 technical_invalid/unqualified**다. 정상 manifest나 독립 baseline 감사는 없으며 자연 보행 완료가 아니다.

| 항목 | 실제 기록 |
|---|---|
| 기준 모델 실행 | 46.640초 / exit1 / original1회 |
| 알려진 실행량 | constructor1, settling2000, neural setup320/capture160, controls99/neural1980/physics4950/rows100 |
| 생산자 q RMS / max | 0.01772869163371381 / 0.0396360246919627 |
| 생산자 tendon RMS / max | 0.001363737833496342 / 0.013190561316255553 |
| 첫 오류 | neutral-actual-imports: actual neutral body imported source closure |
| 부분 증거 | command/tendon100 rows, native4950, trace100, progress, transfer, actual123 imports/15 mismatches |
| 종료 | 모든 owned PID 종료, NVML thread CLOSED/errors0, 예약·jobs0 |
| 새 코드 검증 | whole22 cases/66 phases3.316초, 별도 real fixture1.633초, currentQ fsync/close/reread |
| 누적 비용 | 원본10020.115960899504/10200, 공동126.361/300, 생성110.9570156/150,42 debits |

ROOT 설계→실제 Astra6/max→ROOT 반영→실제 Sol6.1/high FIRST→ROOT FIRST→SAME Sol ONE R1→ROOT SECOND→ROOT 잔여 수정 순서를 수행했다. Ruff 실패0.563/0.566, final0.559와 historical suite cost ID 충돌 전 준비0.781의 비용·실패도 보존했다. 과학·native·trial·E267/E346 및 이전 E358 source/Q는 변경하지 않았다.

15개 누락은 common_descent_body→common_descent_direction 하위 프로젝트 의존성이다. 현재15개 source bytes는 실행 당시 기록 SHA와 같지만 사전 등록이 없었던 사실을 소급 수정하지 않는다. E359 currentQ/원본 실행의 소스 snapshot과 machine report `verification/banc-natural-gait-m3-20260919/parent-e359-root-closed-result.json`을 보존했다. producer startup PASS는 독립 감사 결과가 아니다. missing theta/selected/network-tick raw와 원본 publication binding의 한계를 유지한다.

다음은 [E360 소스 의존성·보존 raw 진단](BANC_M3_E360_SOURCE_CLOSURE_RAW_DIAGNOSTIC_PLAN_20261003.md)이다. 새 소스 사전 검사와 독립 raw-only 프로세스로 제한된 진단을 수행하고 같은 기준 body를 다시 만들지 않는다. E359 판정/F1–F8/M2/UI/F6/과거 비용·실패를 그대로 두며, 이후 startup-constrained neural update 및 실제1.5s/10s·자연 보행은 별도 단계다.
