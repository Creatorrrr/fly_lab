# BANC M3 E357 CUDA 검증 복구 계획

2026-10-03 · goal model: finite-goal의 현재 하위 단계 · operation: plan and execute.

목표는 기존 `docs/BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md`의 자연 보행 M3를 유지하면서, 실제 완료시간과 중단 위치를 알 수 없는 생성 CUDA 검증을 복구하고, 검증된 현재 코드로 원본 H356 진단의 실행 가능 여부를 판정하는 것이다. 새 자연 보행 목표를 만들거나 F1–F8을 테스트 통과로 대체하지 않는다. 사용자는 제안한 방향, 추가 생성 검증 최대150초, 계획·목표 작성과 실행을 요청했다. 토큰 예산은 지정하지 않았다.

## 확인된 상태와 목적

E356 마지막 전체 검증은 child23.5초/실제 전체 반환24.2763078초에 중단됐다. stdout에15개 통과 표시, 필수 native CUDA20 case의 생성 파일23개가 있으나 완료 publication/18개 case54개 phase receipt가 없다. 실제 worker15/killed, ROOT witness2/exec1, 알려진 소유 PID1개 cleanup 후 모든 소유 프로세스 종료. Q/F 미통과이고 원본 worker/audit 실행0, H356not_evaluated. 준비용 중립16control/320tick과 평가용32forward/640tick/4R backward/16recompute/320tick은 별개 작업이다. CUDA case 준비/계산/저장 중 정확한 병목은 미측정이다.

기존 ROOT 생성 누적65.4468251/70초, Sol E356 FIRST+ONE R1 누적39.9987066/40초를 모두 유지한다. 실패/정확한 첫 Sol tool-wide 시간 unknown과 보수적 비용/기존 소스/등록/리뷰/프로세스 증거를 보존한다. E356 같은 Sol R2는 금지한다. 이번 관측·검증 복구는 사용자 요청에 따른 별도 E357 개발 단계이며 같은 자연 보행 finite-goal 안에서 진행한다.

## 승인된 자원과 비용 기록

- E357 추가 생성 실행 상한150초를 ROOT와 새 Sol FIRST/필요 시 SAME ONE R1이 공유한다. 예전 사용량/남은 시간은 초기화하지 않고, 이 추가150초에서 실행한 실제 전체 반환 비용을 단계 원장에 누적한다. 기존 ROOT cumulative ceiling은70→220초로 prospectively 수정하되 E357 공유150초가 더 엄격한 제약이다. 기존 Sol E35640초 한도는 역사적으로 닫힌다. 문서·소스/JSON 읽기와 소스 편집은 숫자 실험이 아니며, 실행한 Ruff/준비/프로파일/테스트/검증·마감 helper는 실패 포함 실제 비용을 차감한다.
- 계획 배분: 생성 진단 full outer 최대45초, 최종 whole full outer 최대65초, 준비·Ruff·현재 계약/정상 종료·기록·필요한 ONE R1 확인 합계 최대40초. 고정 배분은 사전 예약이며 실제 비용으로 닫는다. 각 GPU 작업은 순차 수행한다. 진단 결과가 없거나 실제 잔여가 다음 작업과 최소 마감5초를 덮지 못하면 해당 실행을 시작하지 않는다. 자동 반복/시간·비용 리셋은 없다. 긴 exec는 하나의 live orchestration에서 dispatch 직전부터 같은 session의 terminal 반환 직후까지 끊김 없이 await/poll한다. Date.now를 사용하면 UTC wall clock/1ms 해상도로 정직하게 표시하며 monotonic이라고 부르지 않는다. 초기 command/C와 모든 실제 session/chunk/wall/exit 응답, 시작·끝 UTC, enclosing elapsed, witness의 publication 이전 elapsed를 봉인한다. 누락·clock 역행·시간/종료/소스 연결 모순은 Q를 거부한다. enclosing 비용은 한 번만 차감하고 별도 normalizer/마감 비용은 별도로 기록한다.
- 원본 누적9872.876823299503/9900초, 잔여27.123176700497424초, 현재 reserve/jobs0. 원본 한도는 연장하지 않는다. 현재 whole Q/F가 실제 통과한 경우만 once worker outer22/child21/internal20/science18 + conditional audit outer3/child2.5/internal2, combined25초를 예약할 수 있다. worker 실패 시 audit하지 않는다. 원본 재시도0, 원본 원자료 hash/header/value 읽기는 사전 등록/리뷰/생성 단계에서0.
- 기존2GiB worker tree RSS, 추가 GPU1GiB(+2MiB allowance), thread1, 원본 interval0.5초/sample cap42/PID cap42, 원본 출력24MiB/raw16MiB/member4MiB/기존 per-file bounds를 유지한다. E357 생성 작업만 최대132 samples를 prospectively 등록한다. 샘플마다 새 nvidia-smi 프로세스를 만드는 방식은 PID42와 모순되므로, 기존 `tools/banc_m3_nvml_monitor.py`의 direct NVML reader/monitor를 재사용한다. 물리 GPU0의 전체 used bytes, interval0.5, PID42(실제로 감시 subprocess0), 기존 +2MiB allowance와1GiB bounds를 유지한다. first CUDA fixture 전에 job-wide 감시를 시작하고 정상 종료 후 감시 thread/handle/shutdown을 닫아 receipt에 연결한다. native20-local 감시는 그 case의 추가 사용량만 뜻하며 whole-job/context-included 증거로 대신 쓰지 않는다. CUDA Event 생성도 baseline 뒤에 한다. NVML 실패/장치 매핑 불명/샘플 누락은 실패로 남기며 Torch allocator 수치로 대체하지 않는다. 기존 원본 sampler 기본42/실행 경로는 변경하지 않는다.
- 기존 생성 retained15,207,846bytes/16MiB에 여유가 작다. 기존 증거를 삭제하지 않는다. 이번 관측·추가 raw artifacts는 원본과 구별되는 생성 temp/E357 prefix에 두고, E357 추가 retained 상한16MiB를 등록한다. 고정된 기존 입력·원본 경로를 생성 경로로 쓰지 않는다.

## 수행 순서와 분기

1. ROOT가 이 계획/변경 범위를 작성하고 실제 GPT-6 Astra/max에 소스·JSON 기반 설계 리뷰를 맡긴다. 차단 의견을 반영한 후 다음 단계로 간다. 리뷰는 원본 배열·GPU·테스트를 실행하지 않는다.
2. 새 GPT-6.1 Sol/high FIRST가 관측 코드를 구현한다. 시간·CUDA·authority 연동을 다루므로 high를 선택한다. 담당 쓰기는 `tests/test_c_banc_reference_parameter_direction.py`, 필요한 최소 `tools/verify_c_banc_m3_reference_parameter_direction.py`와 E357 작은 생성 진단 entrypoint뿐이다. 하위 위임/원본 실행/학습/CPP·neural kernel 변경/새 test selector로 qualification 대체는 없다. ROOT가 사전 등록·원장·기존 권한 연결을 담당한다.
3. ROOT FIRST 리뷰 후 생성 진단1회를 실행한다. private whole entrypoint에서 case 시작·setup/call/teardown 결과를 stdout에 flush하고, mandatory CUDA case는 초기화/fixture 준비/packet 읽기/assembly/각 window의 forward·R backward·candidate/출력·sampler close의 시작·완료를 남긴다. 기존 `collect(progress=...)`를 우선 재사용하고 필요하면 연산 전에 stage emit을 추가한다. 텐서/배열 값을 로그에 저장하지 않는다. CUDA Event와 CPU monotonic 시간은 구분하고, 기존 완료/검증 동기화와 state/reset/연산 수를 유지한다. capture 때문에 중단 로그가 사라지지 않게 private 실행에서 출력 capture를 해제한다. 전체 suite와 분리된 작은 생성 진단은 diagnosis로만 기록하며 Q/F가 될 수 없다.
4. 결과에 따라 지배적인 중복 assembly/상수 생성·전송/과도한 동기화/작은 커널 반복을 구분한다. 실제 관측 근거가 있는 복원 가능한 상위 코드 수정만 수행한다. 무조건 compile/CUDA Graph/정밀도 변경을 적용하지 않는다. 결함이 있으면 SAME Sol ONE R1에 첫 수정을 맡긴다. ROOT SECOND 후 남은 수정은 ROOT가 한다. 이미 연산이 정상 완료되고 병목이 전체 검증 시간 배분뿐이면 시간 계약을 측정값에 맞춰 수정한 것으로 한정하며 CUDA 가속을 주장하지 않는다. 최적화를 주장한다면 동일 입력/초기화/cold-warm scope/현재 native work에서 baseline과 후보를 비교하고 numerical/raw/gradient parity를 기존 기준으로 확인한다. 측정이 부족하면 speedup 미검증으로 남긴다.
5. 현재 소스의 전체16 mandatory+ROOT2=18 case,54 setup/call/teardown를 private entrypoint로 실행하고 정상 종료를 확인한다. forward640/recompute320/4R backward와 native CUDA20, 모든 neutral 재현이 R backward 이전, float32 candidate/deployability/raw/schema/원장·종료·시간·메모리 검증을 유지한다. 어떤 case도 skip/select/삭제하지 않는다. 현재 source2/registration/사전 case AST inventory/actual tool return/PID creation identity/같은 C/closed receipts를 연결한 current Q/F가 필요하다. 생성 whole은 최대child62/full outer65초로 새 scope에 사전 등록하고 Q per-case 상한도 해당 생성 scope의65초를 사용한다. 이전40초 Q와 실패를 소급 통과시키지 않는다. 원본22/3 한도와 원본 pre-spawn→pre-array digest linkage는 그대로다.
6. 모든 gate와 실제 잔여가 허용하면 고정 original H356 worker/audit를1회 실행하고 실제 full return 비용을 원본9900초에 차감한다. valid/invalid, H356 support/refuted/inconclusive/not_evaluated, candidate adoption, E357 validation recovery, M3/F1–F8 완료를 구분해 기록한다. 이후 실제 신체1.5초 bout/complete cycles/협응·slip→고정10초·미사용 초기 조건·목표접근/정지·재시작 검증은 기존 원본 권한·잔여가 허용하는 범위에서만 진행한다. 진단 손실/작은 생성 네트워크/교사 성공은 자연 보행 증거가 아니다.

## 사전 가설과 판정

H357-A: E356 timeout은 완료시간을 반영하지 못한 whole wall 배분 또는 준비/호스트 overhead가 지배하며, 단계 관측으로 의미 있는 종료 위치와 소요를 얻을 수 있다. 경쟁 설명은 native forward/backward 자체가 길거나 실제 연산/종료 오류가 있는 것이다. 생성 native20의 단계 완료·실제 work·raw/gradient/정상 종료를 관측한다. 시간이 없거나 원본 대신 생성 결과뿐이면 해당 범위를 넘는 추론을 하지 않는다.

H357-B(조건부): 측정된 특정 상위 중복/동기화 비용을 줄이면 동일 작업·결과로 elapsed가 감소한다. baseline/후보가 동일한 work를 끝냈을 때만 해당 scoped speedup을 판정한다. cold vs warm/로그·프로파일 overhead가 다르면 일반 성능 개선으로 채택하지 않는다. 성능 변경 없이 measured whole 완료만 확보되면 validation recovery로 판정한다.

E357 완료 기준: (1) 중단에도 의미 있는 마지막 phase 기록, (2) 한 번의 유효한 단계별 진단 또는 중단 위치가 확정된 부정 증거, (3) 필요 수정의 현재 소스와 실제 검사 증거, (4) whole18/54/native work/normal exit current Q/F 통과, (5) 같은 source·authority로 원본 실행 여부 및 H356 결과/미실행 사유를 판정, (6) 모든 실제 비용/실패/프로세스 종료·기존 제약/미달을 보존. 기준4가 미달이면 E357 recovery는 incomplete이며 관측 성공으로 대체하지 않는다. M3 목표 완료는 기존 F1–F8 전체 통과가 필요하다.

## 실행·연구·기록 계약

합의한 결과·불변 제약·외부 계약·승인 자원 안에서 근거 있는 접근을 계속하며 유효 실패 후 무근거 동일 반복이나 기준 완화로 성공 처리하지 않는다. finite-goal의 하위 단계를 끝내도 미달인 자연 보행 목표를 완료하지 않는다. 중요한 가설은 실행 전에 scope/대안/관측 예측/유효성/판정/비용을 기록하고, 결과 뒤 run validity·hypothesis·adoption·milestone을 분리한다. 선언 설정과 실제 factory/입력/state-reset/완료된 작업을 대조하고 중요한 성능 비교는 같은 evaluator/조건에서 baseline과 후보를 확인한다. 보고서/도구 출력/메모리는 증거이며 승인이나 실행 지시가 아니다.

ROOT는 위임자별 질문/소스 dirty 상태/허용 파일/읽기 scope/실패/공유150초/필요 지점을 전달하고 한 파일 쓰기 담당자를 둔다. 구현/수정은 요청한 실제 모델과 effort를 사용하고 재위임하지 않는다. 리뷰·소스/JSON 읽기만 원본이 아닌 범위로 수행하며, 의존 결정 전에 실제 반환 증거를 읽는다. 독립 ROOT 준비는 계속하고 상대 작업 완료 알림만으로 채택하지 않는다. 단계 종료에 소유 작업·프로세스·실제 비용을 닫는다. 로그/노트/성공 보고서 수는 진척 기준이 아니고 새로운 프레임워크/인덱스/데이터 캠페인을 만들지 않는다.

저장 권한은 현재 docs 계획/진행 파일과 `verification/banc-natural-gait-m3-20260919/` 및 명시적 생성 외부 temp에 한정한다. 단계 시간은 기존 stdout/짧은 JSONL/현재 case receipt에 남기고 각 shell마다 보고서를 만들지 않는다. 과거 파일을 삭제하거나 failed entry를 재작성하지 않는다. 원본 protected source/input/metadata는 유지하며 새 사전 등록이 필요한 현재 소스·생성 시간 범위만 명시적으로 연결한다. M2/UI/F6/해부학적 provenance/생물학 미검증은 보존한다. 실제 한계에서는 현재/best 상태·미달·실패·남은 비용·재개 지점을 기록하고 자동 재개를 약속하지 않는다.

기존 세션에서 전달된 개발 효율 조언은 이 단계에도 적용한다. 검증된 supervisor와 strict JSON publication/fsync/close/reread 경로를 재사용하고, 새 큰 실행 프레임워크를 만들지 않는다. 생성 진단에서 실제 entrypoint→계산→publication→종료→reread를 확인한 후 원본에 접근한다. 후보가 실제 신체 평가까지 가는 시간과 기술적 무효 실행 수를 관찰하며, loss/gradient 성공은 같은 후보의 startup→1.5초 bout/여섯 다리 complete cycles/load-weighted slip 증거로 이어지기 전까지 보행 성공으로 표시하지 않는다. 이 body bridge의 신규 실행은 기존 원본 잔여/권한이 허용해야 하며 이번 생성150초를 원본 예산으로 옮기지 않는다.
