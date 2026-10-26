# E358: 후보 저장에서 동일 후보의 실제 보행까지

2026-10-03 09:58 KST. `goal-planner` finite-goal / plan-and-execute 적용. 사용자의 최신 “제안한 추천 방향으로 계획·목표 작성하고 진행” 요청은 원본 추가300초와 근거 있는 변경 후 최대1회 재실행을 승인한다. 전체 목표는 [M3 F1–F8](BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md)이며 이 진단 단계를 완료했다고 전체 목표를 완료하지 않는다.

## 결과와 현재 근거

감각→전체 BANC 신경→근육 디코더→물리 몸체의 동일 후보 경로를 실행한다. 외부 CPG/위상/교사 관절값/root 보정은 최종 경로에 넣지 않는다. 해부학적 topology, 부호, 원본 계수·디코더·native f32 계산과 모든 F1–F8은 유지한다. 기존 M2, UI, F6 노출, 실패와 누적 비용을 보존한다.

E357 현재 생성 검증은18 cases/54 phases 및 CUDA20 work를 완료했다. 원본 실행20.8781376초는 중립4개 재현·R 역전파4개 반환 뒤 science18 deadline에서 끝났다. 후보 계산·평가·저장·독립 감사·실신체 보행은 없다. 그러므로 CUDA 가속이나 자연 보행의 성공으로 해석하지 않는다. 기존 source와 Q/F를 불변으로 유지하고 새로운 경계만 별도 검증한다.

읽기 조사에서 실제 handoff API는 GraphStore/StateStore→SSPRecoveryResourceTrial.from_checkpoint→SSPEfficacyTrial(positions, groups, theta_actual)→BancWalkingSession.advance였다. 기존 E346 도구는 E343 raw field, palette15461, 후보 hash, output과 resource limits를 고정한다. 이 도구를 monkeypatch하거나 이전 후보를 대입하지 않는다. 순수 검증·관측 함수 및 trial API만 재사용한다. E346의 conservative80초 비용과 science 미독립 자격 정정, E353 teacher-only 결과의 자격 제한도 유지한다.

## 설계

새 `tools/verify_c_banc_m3_candidate_body_handoff.py`와 해당 tests를 구현한다. 기존 reference-parameter-direction, coordination core, neural/body/trial, E267/E346 감독 소스와 기존 테스트는 수정하지 않는다. ROOT가 계획·자원·실험 및 최종 계약을 기록하고, Astra6/max가 설계·실제 소스를 리뷰한 뒤 Sol6.1/high가 구현한다. 수학을 바꾸지 않는 여러 모듈 연동 및 수명·저장 계약이므로 high를 사용한다. ROOT FIRST→같은 Sol ONE R1→ROOT SECOND→ROOT 잔여 수정 순서를 지킨다.

실제 Astra6/max가 M1–M5 반영된 계획SHA07b174dac5b7847fb7f2845cda7fd619428458bf25c23ec89e50c8d45045ce47을 재독하고 Sol FIRST 진행 가능으로 판정했다. 상세 구현 acceptance는 `verification/banc-natural-gait-m3-20260919/parent-e358-astra-design-review.md`에 따른다. ROOT는 모든 필수 지적을 수용한다. 실제 recovery API 파일은 `tools/banc_m3_ssp_recovery_trial.py`이다. GPU 조건은 **sampled delta +2MiB ≤1GiB(candidate) /8GiB(body)**이며 cap에 allowance를 더하지 않는다. 생성 실행·원본 science/array/body 실행0으로 리뷰 완료. 이 리뷰는 원본 실행 자격이 아니다.

새 도구의 첫 원본 worker는 불변 E357 라이브러리의 기존 packet loader, cadence label loader, collect, make_direction 및 publish_generated를 호출한다. 완전한 현재18-case Q/F와 immutable source pins를 pre-spawn/pre-array에서 확인한다. 새 adapter generated 자격은 original0, 명령/자원·digest·publication·후보 선택·잘못된 transfer 거부를 검증한다. E357 과학 소스 불변을 확인하므로 같은18-case CUDA work를 중복 실행하지 않는다. adapter가 해당 소스를 바꾸면 이 재사용은 무효이며 전체 재검증이 필요하다.

pre-Torch NVML physical0 baseline, 실제 device/UUID, 첫 오류 및 shutdown1·thread close를 확인한다. candidate와 body는 각각 기존 메모리 한도를 유지한다. NVML subprocess0; 오래된42-PID sampler 제한을 늘리는 대신 직접 모니터를 재사용한다. 추가 CUDA synchronization, dtype/kernel/reset/window/gradient 변경은 없다. 확인 지점의 CPU phase elapsed와 기존 CUDA confirmation 직후 timing만 기록하며, 현재 총 work32 forward/640 ticks,16 recompute/320 ticks,4 R backward와 all-neutral-before-gradient를 유지한다. phase timings는 다음 변경의 근거이고 matched speedup 증거는 아니다.

worker의 fresh output은 E358 attempt-specific 경로이며 E356 failure를 덮어쓰지 않는다. 새 원본 audit는 current raw/manifest/selected-ledger를 읽어 불변 analyze·array digest·selected relationship을 다시 확인한다. branch normal, nonzero 실제 f32 direction, 원본 sign/topology/unselected bytes, palette≤16384, source identity를 만족해야 handoff할 수 있다. R/Q_exact/T 개선 여부를 기록한다. eligible 후보만 body에 넣고 기술적 성공을 gait 성공으로 승격하지 않는다.

리뷰 반영 M1/M2: E357의 original validate_supervisor_contract는9900초/이전 paths/limits를 고정하므로 새 권한 검사로 재사용하지 않는다. E358 parent-owned 계약을 **같은 parent ENTRY의 실제 child Popen 전**과 child의 library/native/array 읽기 전에 재검사하며 exact argv·stage/path·source closure·현재 adapter Q·E357 science Q/F·예약·cumulative 값을 결합한다. 부모는 원본 배열 읽기0/Torch-native-body0을 확인한다. 불변 collect/publish 결과의 old library source2 provenance를 그대로 두고 E358 source·계약·예산·actual full exit envelope를 별도로 붙인다. body admission은 독립 감사된 branch normal 및 literal `analysis.H356 == conditional_support`만이다. 이전 후보/E356 실패/Q만으로 admission하지 않는다.

body는 saved control0 **whole body**를 원래122 array leaves와 state/manifest의 유한 roster로 복원한다. E318 신경 window 또는 교사 body로 대체하지 않는다. body input은 E345 graph/whole/recurrent_mask/startup/reference metadata의 명시된 역할과 현재 E358 audited candidate만이다. E343 candidate 입력은 제외한다. 입력 bytes/hash/header/value 검증은 감독된 원본 worker 안에서만 한다. 생성 fixture와 원본을 구분한다.

실제 efficacy trial을 만든 후 inverse whole exact, graph/source decoder identities, candidate CSR 선택 weight=f32(f64(source32)*theta_actual[group]), 모든 unselected bytes·topology/sign, packed palette/row/column/pointer guards를 확인한다. palette는 현재 후보로 계산하며 E343 숫자를 재사용하지 않는다. 실제 감각 feedback을 통해300 controls/6000 neural ticks/15000 native physics steps(1.5s)를 진행한다. 기존 constructor의1 body/2000 settling/320 setup/160 capture work를 요청·반환·device-confirmation으로 기록한다. 실제 native observer contact/load/slip/upright·clock 및 CPG digest를 확인한다.

99 controls에서 기존 startup RMS/max(.018/.04 rad; tendon .009/.025) 게이트를 유지하며 실패 즉시 중단한다. 1.5s 완료 시 기존 물리 screen forward≥1.6790085779712631mm, supported slip≤1.8026436968249955mm/s, 연속 전진 bout≥.3s, 여섯 다리 각각 complete cycle>0, native health를 요구한다. E346 old smooth/shape 비교는 별도 진단으로 보존하되 R 기반 E358 후보를 old S 목적 성공으로 표현하지 않는다. 이 screen은 고정 F3–F5의10s 판정이 아니다. 통과한 동일 후보에서 잔여 시간과 측정 비용이 충족될 때만10s/미사용 초기상태/목표·정지·재시작/차단·저장 F1–F8 확인으로 진행한다. 자원이 부족하면 실제 미수행과 필요한 최소 비용을 기록한다.

리뷰 반영 M3/M4/M5: 기존 E346 lifecycle의 original factory/output/clock와 science60/internal72/supervisor80 및 publication를 새 original gate로 우회하지 않는다. 새 body lifecycle가 요청·host 반환·device 확인·unknown work, 초기 whole inverse·boundary command/tendon exact, observer detach→body close, close-time 오류와 provisional/revocation을 명시한다. 직접 NVML의 기본 interval.1/qualified8GiB에 의존하지 않고 명시적으로 interval.5·sample count·physical mapping·후보1GiB/body8GiB를 확인한다. body worker full exit 뒤 **독립 body readback audit**가 literal manifest/닫힌 source와 transfer linkage, raw clocks/typed counters/native contact/trace의 startup·물리 screen을 재계산하고 동일 후보 theta/selected byte identity를 확인한다. 이 감사와 parent full-return/owned PID 종료가 없으면 screen은 unqualified이다. 독립 감사는 원본 body를 재구성하거나 physics/neural을 다시 실행하지 않으며 array read/time은 별도 원본 비용이다.

## 실행 전 실험 계약

H358a: 과학 계산을 바꾸지 않은 새 wall/monitor 계약으로 하나의 실제 f32 후보 계산·평가·저장·독립 감사까지 완료할 수 있다. 대립 설명은 계산보다 준비/저장/프로세스 종료가 제한을 소비하거나 후보의 f32/palette/response 게이트가 실패하는 것이다. source/current Q/pins/actual returned work·strict JSON·fsync/close/reread·owned PID creation-time 종료가 유효성 조건이다. 유효한 계산·감사 완료는 H358a support이며, gate 실패는 해당 한정된 후보에 대한 reject; timeout은 보행 가설의 반증이 아니다.

H358b: H358a의 동일 후보가 startup과1.5s physical screen을 통과한다. teacher/raw response/saved old candidate는 비교 증거로 대체하지 않는다. 실제 body work·sensory feedback·contact/trace/transfer 및 자원/폐기 종료가 유효성 조건이다. screen 통과/실패/미수행을 구분하고 모든 고정 F1–F8을 그대로 남긴다. 다음 후보 선택에 노출한 초기상태/검증은 F6 미사용이라고 재명명하지 않는다.

첫 시도 결과 후 **최대1회** 재실행은 근거에 따른 변경이 있어야 한다. 같은 timeout 설정의 반복은 없다. 첫 시도에서 science45가 동일 실제 backward/candidate 단계에서 제한되면 기록한 work/phase를 근거로 두 번째 science65/outer75 계약을 사용할 수 있다. f32/palette/R response가 실패하면 변경 요인을 먼저 명시하고 별도 ROOT 설계→Astra 리뷰→Sol 단계로 반영하며 잔여 원본/생성 자원을 초과하지 않는다. 학습률/수학 변경은 자동 허용된 동일 source qualification으로 취급하지 않는다.

## 공유 자원과 종료 계약

- 원본 기존9893.754960899503초를 유지하고 승인으로 한도9900→10200초. 잔여306.2450391004968초 중 **E358 stage 자체는300초 이내**, 기존6.245초는 headroom이다. 전체 exec full-return, 준비·감사·실패·저장·마감 helper를 한 번만 debit한다. 별도 원본 preflight/profiling/비공개 array 읽기는 없다.
- 첫 candidate outer55/child52/internal50/science45초, audit outer8/child7/internal6.5/science6초. 최대1회 변경 retry outer75/child72/internal70/science65초와 audit8초. body outer130/child125/internal120/science115초, body audit outer12/child11/internal10/science9초. 전부 사전 예약하면288초이며 마감 여유12초와 기존headroom6.245초를 보존한다. 실제 사용만 debit·unused reserve release 후 남은 비용으로 다음 고정 gate가 가능한지 판단한다. 실행 가능성이 없으면 시작하지 않는다.
- 기존 생성 공유150초의 이미 사용80.1900156초를 유지한다. 남은69.8099844초를 새 adapter 최소 검증/준비·자격/마감에 사용한다. 새 생성 시간 승인이나 기존 비용 리셋은 없다. 기존 ROOT65.4468251/70, Sol39.9987066/40 기록도 유지한다. 모든 새 numeric import/test/format helper는 공유 debit, 소스·JSON/MD 읽기와 편집은0초이다.
- 후보 host tree2GiB·additional GPU1GiB+2MiB allowance, body는 기존 E346 host6GiB·additional GPU8GiB(physical device total 이내). CPU thread1, GPU physical0. sample cap candidate132/retry172/body272, NVML subprocess0, 직접 관측된 owned tree만 종료한다. RUN 오류·OOM·cleanup kill·thread/identity 누락은 자격 없음.
- 기존 E357 retained16MiB 전부 보존. E358 새 생성 단계는 기존 per-stage16MiB, 원본 candidate 각각24MiB/body32MiB/audit1MiB cap 및 예상 실패·마감 공간을 advance reservation에 포함한다. 오래된 실패 삭제나 cap 회피 재분류는 없다. 새 stage metadata도 경계 안에 집계한다.
- 출판은 strict JSON allow_nan=False, NumPy scalar/tuple normalization, required cumulative fields, 파일 flush/fsync/close/reread, 첫/secondary 오류 보존 및 provisional 출판 후 revocation을 적용한다. parent 실제 full-return과 owned creation identity closure가 확정되기 전 과학 자격·승격은 provisional이다.

## 목표·실행 상태

기존 API의 full M3 goal은 마지막 조회에서 blocked였다. 최신 사용자 요청으로 이 승인 범위의 작업을 재개하되 API resume 도구는 없으므로 실제 조회 전 active라고 쓰지 않는다. 새 차단 감사는0부터 시작하며 과거3회 audit를 보존한다. 별도 중복 unfinished goal을 만들지 않는다. run validity / H verdict / adoption / F1–F8 milestone / actual API status를 분리한다. 자연 보행·생물학적 정확성은 직접 결과 없이는 미검증이다.
