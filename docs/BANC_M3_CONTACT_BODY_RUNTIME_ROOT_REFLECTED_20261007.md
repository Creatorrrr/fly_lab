# 실제 신체 실행 시간 진단 — ROOT 리뷰 반영, 2026-10-07

사용자의 추가 승인 없는 자율 진행과 기존 finite-goal을 유지한다. 목표는 고정된 BANC 후보의 실제 신체 실행에서 유효한 startup/short 결과를 얻는 것이며, 이번 계측은 그 결과를 위한 진단이다. F1–F8, BANC provenance, 외부 CPG·교사·위상·궤적·root 보정 금지, M2/UI와 미사용 F6, 모든 이전 실패·비용은 바꾸지 않는다. Goal ACTIVE/INCOMPLETE, 후보 미채택, biology UNPROVEN, exact native VJP UNVERIFIED다.

## 실제 설계 리뷰와 ROOT 결정

실제 GPT-6 Astra/max 리뷰 JSON25429B/SHA d7796c6770f89bb3693182c9ecd4020e1e9d3e098e07b065151f1d5236a92824와 MD11578B/SHA263007ef5998ac7dece6b7e7cf52ac620de86dbc840b4c8ab643f130cab31406을 전부 읽었다. 리뷰의 Source/작은 metadata 32개 핀을 현재 파일과 대조했다. 최초 핀 확인 명령의 slash 경로 비교 오류는 읽기만 수행한 행정 오류로, 정규화 후 모두 일치했다. Original 입력·Python·테스트·CUDA·body 실행은 하지 않았다.

원 설계는6677B/SHA10495252cf94bac127e65dae71547fee99247c586fbf52f4f5803689591232e8 그대로 보존한다. Astra의 B1–B5를 모두 채택한다. FIRST는 phase timing과 acquired failure journal에 한정하고 proof cache/CUDA/warmup/constructor/candidate/control/evaluator 변경은 측정 이후 별도 분기로 유보한다. 안전성·과거 기록을 가로지르는 작업이므로 실제 GPT-6.1 Sol/max FIRST를 사용한다. ROOT FIRST→필요할 때 같은 Sol의 sole R1→ROOT SECOND→ROOT 잔여 순서다.

Actual c의268.84초 도구 full-return과270.0479967999272초 owner interval을 별개로 보존한다. 내부 ENTRY deadline 실패이고 outer timeout=false이며 once debit/closed/양쪽 EOF가 확인됐다. Durable result0/work=null, constructor/neural/physics0 또는 보행 수치를 만들지 않는다. arm_data와 continuous_arm 중 정확한 병목은 UNKNOWN다. Qualification-x의 역사적 scope 문구는 t/48·49·50으로 남아 있으나 실제 structured pins와 JUnit은 same-x60·61·62/11·24 PASS다. 그 바이트를 수정하거나 새 Source에 재사용하지 않는다. 원 설계의 closed276/158448154B 표기는 실제276 debits/158 calls,448154B로 바로잡는다.

## 계측 계약

새 journal schema는 flylab.m3.contact-body-runtime.journal.v1로 고정한다. 실제 owner perf_counter ENTRY를 원점으로 사용한다. phase는 admission/body entry, body-plan, candidate-header/load/hash, arm-data, pre-construct gate, construct, initial-transfer, control-loop, posthash, science-publication, failure-metadata-publication이다. START와 실제 RETURN/FAIL을 구별하며 현재 미반환 phase와 관측 시각을 기록한다. 독립된 journal 객체가 result/held_report dictionary 교체 이후에도 살아야 한다.

최대 phase events64, 전체 snapshot JSON32768 bytes, error message1024 characters, secondary errors8개다. 최초 오류는 write-once, 마지막 active phase/terminal은 예약된 공간에 보존한다. Overflow는 events_dropped/bytes_truncated와 불완전성을 명시하며 누락을 미실행으로 해석하지 않는다. 실제 기존 counters의 작은 scalar projection만 사용할 수 있고 array/state/graph/result 전체를 순회하지 않는다. Fence label별 attempted/completed/failed와 누적 wall, 실패·진행 중 부분 시간을 집계한다. Phase의 inclusive wall과 nested fence wall은 따로 보고하며 합산하지 않는다. 계측을 위한 추가 GPU 동기화는 없다.

긴 opaque 단계의 START는 기존 owner가 소유한 stderr stream으로 미리 보낸다. 이 경로는 고정된 call/role/Source/request identity를 가진 진단 metadata이며 유효성 attestation이나 science result가 아니다. 별도 파일 경로 권한을 만들지 않는다. 기존 owner는 이미 받은 raw stderr를 기존 종료·EOF 흐름으로 보존한다. Worker의 emit은 최대16회, 각32768 bytes 안에서 cumulative524288 bytes 이내이며 기존 stderr1MiB/metadata64KiB/combined96MiB를 유지한다. Terminal emission을 위해 최소 한 번의 slot과 bytes를 예약한다. Tick별 출력은 없고 control-loop는 기존 requested/host-returned/device-confirmed counters의 집계만 남긴다. 마지막 START가 hard Job kill 이후에도 이미 받은 stream에 남아야 한다.

science 권한은 actual owner ENTRY+270에 끝난다. 이후 새로운 Original read/header/hash/posthash, constructor/neural/physics, Source/history 재검증 scan 또는 science 재진입은 금지다. 이미 취득한 작은 metadata를 포장하는 단회 terminal emit만 actual ENTRY+285 전에 제한하며 새 clock origin이나 Admission deadline 변경을 쓰지 않는다. Emission은 더 작은 고정 flush allowance 안에서 시작·반환 시간을 검사하고 여유가 없으면 명시적으로 거부한다. Blocking pipe write가 끝나지 않으면 기존 owner의285 cutoff가 Job을 끝내며 해당 emission은 NOT_RETURNED다.285→300은 owner termination/EOF/outer publication 전용이다.

모든 기존 science 경로의 cold Source/input/header/hash/owner/PID creation/named Job/token/lease/shared lock/reservation/output 검사는 유지한다. Journal channel은 그 검사를 통과하는 수단이 아니며 run_valid=true, Source integrity/posthash success, science counters를 새로 만들 수 없다. 소유권 guard가 실패했거나 과학 deadline 후 fresh 검증을 할 수 없으면 기록은 untrusted diagnostic metadata로 표시한다. 실제 권한·EOF·known-ended의 최종 증거는 기존 outer다. 다른 요청/프로세스/이름의 파일 생성, late science, 재시도 루프는 없다. 기존 Publisher.write의 deadline cold fence를 failure channel에 재사용하지 않는다.

## 기존 자원 내 재배정과 행정 순서

Closed276448154B/SHA1ddc9e0c8065914918e9bcbdd0617932359bc5ae1b313ead35ed888df025e351에서 generated822.9329999999999/900/shared remaining148.85498440000015, owners/pending/holds/reservations0를 재확인했다. Original12263.084960899505/16200/stage2369.33/6300/science2242.969/6000/remaining3936.9150391004946와 모든 한도는 불변이다. root GOAL_PLAN151552B/SHA16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace도 불변이다.

Source 구현과 paid checks 전에 ROOT의 metadata-only 절차로 distinct conserved60 event를 등록한다. 기존 Source-x의 Native generated-budget 함수와 metadata guard를 그대로 추출한 별도 행정 script로 현재900의 초기 policy/네 event/ordered typed debit sum/전부 charged/closed/lock 부재를 검증한다. 새 schema flylab.m3.contact-body-runtime.generated-transfer.v1, transfer_id contact-body-runtime-20261007-60, from900/to960/transfer60/new_resource_grant0, 정확한 closed276 및 초기 policy/기존 네 event/기존 직접 사용자 autonomy 핀을 담는다. CreateNew event 후 동일한 live bytes와 모든 원장 필드가 unchanged임을 다시 확인하고, 새로운 direct_science_generated_runtime_transfer pointer와 local limit960 두 필드만 atomic ledger replace한다. 이전 원장·event·사용량·비용·Source는 덮어쓰지 않는다. 이 행정 절차는 ROOT만 수행하며 Original payload/과학/테스트 실행은 없다.

새 unused137.06700000000012는 shared148.85498440000015 안이며 fresh lint8+math12+gate80+closing20=120을 수용한다. 재배정은 사용량 debit/공유 credit/grant/reset이 아니다. 이후 Sol은 Python/Native catalog에 정확히 동일한 fifth event와 lower-state/no-duplicate/prefix/conservation 검사를 구현한다. 보호 실행 전 실제 generated 검증으로 두 catalog의 일치를 확인한다. Float64 fold와1700-baseline-stage 순서, per-call120, closing20, 기존 네 event의 source·caps·실패는 바꾸지 않는다. 남은 자원 내에서 실패하면 criteria/assertions를 줄이거나 close reserve를 쓰지 않는다.

## 새 Source와 단일 d 실행

ROOT는 Source-x의 기존12개 행정·Python·test 파일 및 이번 수정 대상인 startup-ray.py/owner execution test를 실제 freeze와 대조한14개 Source archive로 보존했다. index4745B/SHA6d719bb76579b183d3f7c4859aa24898e14f614988214f9ba58fb8c9e9704de2다. 이는 Source 보존이며 새로운 qualification이나 protected run이 아니다.

새 d는 별도 immutable prospective contract와 call contact-learning-20261007-d-body 및 contact-learning-20261007-d-body-audit를 사용한다. Same audited k identity/alpha.0001/index4/theta/selected/full native palette와 기존 init/control300/startup99/short evaluator/모든 physical caps를 유지한다. b/c descriptor·Source·failures·consumed calls 및 unused audits는 그대로 남긴다. d history는 exact closed276/c technical failure, 실제 Source-x qualification/Source-x archive14와 모든 기존 b/c history를 분리하여 검증한다. 자동 반복0, body 최대1, genuine exit0/run_valid=true일 때만 CPU audit 최대1이다. d-body before는 fresh generated63·64·65의3 returns 이후279 debits/161 calls/Original23, audit은 sole d body 후280/162/Original24다. Fixed counts에 임의 extra calls를 끼워 넣지 않는다.

Sol FIRST는 proposal directory에서만 소스를 구현하며 canonical/ledger/contracts/grants/freeze/qualification/admission/request/Original payload를 직접 수정·실행하지 않는다. ROOT가 두 리뷰와 필요한 잔여 수정을 끝낸 뒤 canonical 적용/Source-y freeze/actual lint63·math64·gate65/fresh qualification을 순서대로 맡는다. 모든 기존 math11/gate24 names/classes/assertions, full owner9/metadata3는 보존하고 deadline/journal/overflow/hard-kill 및 새 transfer/history 음성 경로를 기존 planned qualification에 포함한다. 추가 cases가 필요하면 새 expectation에 명시하되 기존 cases를 삭제하지 않는다. 비용은 genuine first FINAL만 once charge하며 unknown return은 hold하고 재실행하지 않는다.

## 결과에 따른 다음 결정

유효한 journal에서 body-plan/arm-data의 cold 검증이 실제 시간을 지배하면 별도 안전한 immutable proof reuse 설계를 검토한다. 그 전에는 cache를 채택하지 않는다. Constructor/CUDA/control 병목이면 그 측정에 맞는 다음 변경을 고른다. 기술적 실패의 journal은 보행 실패의 과학적 판정이나 후보 채택이 아니며 d audit도 수행하지 않는다. 유효한 body와 audit가 실제 startup/short gate를 통과해야10초 F3–F5와 미사용 F6/F7/F8 순서로 이어갈 수 있다.
