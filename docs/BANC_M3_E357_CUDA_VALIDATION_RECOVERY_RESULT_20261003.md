# BANC M3 E357 CUDA 검증 복구 결과

2026-10-03. 생성 CUDA 검증 복구는 완료했다. 자연 보행 M3 목표는 미완료이며, E357 검증 종료 시 기존 finite-goal은 active였다. 목표와 F1–F8을 축소하거나 초기화하지 않았다.

후속 상태: 원본 잔여6.245039100496797초·재시도0·저장 후보 부재라는 같은 차단 조건이 세 goal turn에 걸쳐 유지됐고, 소유47개 creation identity와 구현 에이전트가 모두 종료됐음을 다시 확인했다. 실제 goal API를 blocked로 갱신했다. 원본 실행 예산과 새 실행/재시도 계약의 사전 승인 없이는 후보 평가·저장과 신체 보행 검증을 진행할 수 없다. [차단 감사](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e357-blocked-goal-audit-20261003.json)에 기록했다.

실제 GPT-6 Astra/max 설계 리뷰를 반영한 뒤 GPT-6.1 Sol/high가 구현했다. ROOT 1차 리뷰의 수정은 같은 Sol의 ONE R1으로 처리했고, ROOT 2차 리뷰 뒤 남은 수정과 검증을 마쳤다. E356의 기존 Sol FIRST/R1 비용과 종료 상태는 그대로 보존했다.

## 구현과 실제 검증

- 전체 테스트의 시작과 setup/call/teardown 결과를 즉시 출력하고, CUDA fixture 준비·packet assembly·forward·역전파·후보 계산·저장의 단계 시간을 기록한다. 필요한 CUDA Event 결과는 기존 완료 확인 뒤 읽으며, 관측만을 위한 새 동기화를 추가하지 않았다.
- 생성 테스트에서는 Torch/CUDA 초기화 전에 direct NVML 감시를 시작한다. 물리 GPU0 전체 메모리 사용량을 측정하고 감시 thread와 NVML handle을 정상 종료한다. 원본 워커의 기존 표본·PID·시간·메모리 한도는 바꾸지 않았다.
- 생성 native20 진단은 실제 전체 반환18.836초에 정상 종료했다. fixture8.496초, collect4.270초가 관측됐다. strict JSON 저장·fsync·close·재읽기까지 완료했다. 이는 작은 생성 입력에 대한 진단 증거다.
- 첫 전체 실행은27.476초에17개 통과·1개 실패였다. 기존 ROOT positive fixture의 실제 오차6.05%는 고정 기준5%를 넘었으며 분석기의 refuted 판정이 맞았다. 이 실패를 보존하고, 후속 fixture에는 해당 반례와 별도 양성·음성 사례를 넣었다. 과학 계산과 허용 오차는 유지했다.
- 현재 소스 전체 검증은16개 필수+ROOT2개, 총18개 테스트와54개 phase가 모두 통과했다. 실제 전체 반환27.375초, pytest25.15초였으며 현재 소스·등록·종료 증거가 일치한다. 초기 CUDA context를 포함한 GPU 추가 사용량은336,904,192bytes였다. 모든 소유 프로세스가 종료됐고 강제 cleanup은0이다.

동일 입력과 동일 측정 범위의 이전/이후 성능 비교는 수행하지 않았다. 따라서 CUDA speedup은 미검증이다. 최종 보행의 신경 계산·외부 CPG 금지·해부학적 provenance와 기존 M2/UI/F6 계약은 유지했다.

## 실제 원본 BANC 실행

현재 전체 검증을 통과한 같은 소스로 원본 워커를 허용된1회 실행했다. 네 개 중립 재현을 모두 역전파 전에 확인했고 네 번의 R 역전파가 반환됐다. 이후 고정18초 science deadline에 도달해 exit1로 종료했다. 실제 전체 반환 비용은20.8781376초다.

후보 평가·저장과 후속 audit는 완료되지 않았다. 실제 신체 보행 실행도0이며 H356은 not_evaluated다. 이 실행을 유효한 후보, BANC 자율 보행, 자연 보행, 생물학적 검증 또는 F1–F8 완료로 채택하지 않았다. 모든 소유 작업은 종료됐고 reserve/jobs/retry는0이다.

## 확정 비용과 다음 단계

| 범위 | 실제 누적 | 한도 | 잔여 |
| --- | ---: | ---: | ---: |
| 이번 E357 공유 생성 실행 | 80.1900156초 | 150초 | 69.8099844초 |
| 기존 원본 실행 전체 | 9893.754960899503초 | 9900초 | 6.245039100496797초 |

E357 비용에는 실패한 실행과 마지막 마감 helper0.5622754초를 각각 한 번만 포함했다. 기존 ROOT 생성65.4468251초와 기존 E356 Sol39.9987066초는 보존했다. source/JSON 읽기와 편집은 수치 생성 실행이 아니다. 원본 내부 원장의 짧은 부분 시간을 실제 전체 반환 비용으로 대체하지 않았으며 중복 차감하지 않았다.

다음 개발은 원본 준비·네트워크 조립·역전파·후보 평가·저장의 비용을 분리하고, 관측 근거가 있는 중복 구성·전송을 줄이는 방향이다. 성능 변경은 같은 입력·초기화·완료된 연산과 raw/gradient parity를 비교해야 한다. 원본 재시도0과 현 잔여6.25초로는 새 워커 계약을 실행할 수 없다. 남은 생성 실행 예산을 원본으로 이전하지 않는다.

추가 원본 실행 자원이 명시적으로 허용되면 완전히 평가·저장된 한 후보를 확보한 뒤, **같은 후보의1.5초 실제 신체 보행에서 지속 bout·완전한 주기·다리 협응·미끄러짐을 먼저 검증**한다. 이를 통과해야 기존10초 보행과 미사용 초기 조건·목표 접근·정지·재시작을 포함한 F1–F8로 진행한다. 그 전에는 사용자 화면의 자연 보행 모드로 채택하지 않는다.

계획: [E357 CUDA 검증 복구 계획](C:/Users/ckthd/Dev/Projects/fly_lab/docs/BANC_M3_E357_CUDA_VALIDATION_RECOVERY_PLAN_20261003.md). 확정 판단과 비용: [단계 결정](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e357-final-stage-decision.json), [실행 원장](C:/Users/ckthd/Dev/Projects/fly_lab/verification/banc-natural-gait-m3-20260919/parent-e357-execution-ledger.json).
