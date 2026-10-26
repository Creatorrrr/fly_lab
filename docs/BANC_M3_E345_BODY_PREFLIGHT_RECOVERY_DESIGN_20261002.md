# E345: E344 원본 실행 전 실패의 복구

ROOT가 설계했다. 실제 GPT-6 Astra/max 리뷰를 반영한 뒤 GPT-6.1 Sol/high가 구현하고, ROOT FIRST → 같은 Sol ONE R1 → ROOT SECOND → 남은 ROOT 수정 순서를 유지한다.

## 관측과 실패 보존

E344 현재 생성 자격은 CPU62 PASS/1 host symlink SKIP, synthetic8/301 저장·고정 gait 분석, 실제 generated CUDA20/FlyBody50, 같은 독립 CLI 정상 PASS/6 coherent faults REJECT로 닫았다. ROOT54.965215300020645/60s 및 Sol59.13786539998546/60s는 실패 포함 비용이다. 생성 성공은 원본 body 또는 자연 보행이 아니다.

원본 worker는 gate의 `exact typed scientific whitelist`에서 실패했다. 등록 inventory는 `exact_distinct_file_count=135`이지만 파일 dictionary는131개이고 whole NPY122개는 그대로다. freeze와 inventory/dependencies equality는 참이다. ROOT가 metadata를 분해한 결과 files는 whole124(122NPY+state+manifest)와 graph7만 포함한다. 이미 selected_archives로 등록된 recurrent_mask/reference/audited_candidate/startup 네 NPZ가 files dictionary에서 빠졌다. alias 중복이 아니며 기존 selected descriptor의 path/bytes/SHA를 그대로 whitelist에 추가하여 원래 설계대로135개를 닫아야 한다. 새 데이터 또는 후보를 선택하지 않는다.

E267은 worker가 실패한 후 finally에서 계약의 `cumulative_before_s`가 없어 KeyError를 냈다. supervisor/identity 최종 JSON을 만들지 못했다. 실제 worker stderr, 빈 stdout, observer stderr는 보존한다. 원본 output directory는 없다. Original gate 실패 위치는 입력 payload hash/header/load, CUDA/context, constructor/physics 이전이다. 그 경계를 넘어 실행했다고 주장하지 않는다. 최종 PID identity와 wall 기록은 미확보이므로 정상 종료·정확한 실제 비용은 주장하지 않는다. 보수적으로 body 예약80s 전액을 실패 비용에 합산한다. 미실행 audit12s는 해제한다. E344는 technical_invalid, scientific verdict not_tested, retry0로 닫고 같은 E344 명령을 재시도하지 않는다.

## 수정과 실제 보행 검증

1. 변경되지 않은131개 파일 path/bytes/SHA와122개 whole NPY 목록, selected_archive4의 기존 등록 path/bytes/SHA와 모든 ref coverage를 정적으로 검증한다. 새 inventory v2에 빠진 selected_archive4 descriptor를 합쳐 원래135개 distinct file contract를 충족하고 기존 inventory/registration/freeze/source/log를 보존한다. 각 selected path가 files에 들어 있고 SHA/bytes가 같아야 한다; 외부 payload를 새로 해시하지 않는다. Scientific payload hash/header/value 새 접근은 original worker 전까지0이다.
2. 새 registration을 실제 inventory v2 및 기존 runtime descriptor v2, E345 설계/실제 Astra 리뷰에 연결한다. 같은 후보 E343 theta_minus, whole restore, original source physiology/decoder, actual8 geometry, live sensing→BANC20→physics50, startup99, native partial counters, 모든 고정 gait/slip/bout/cycle gate는 그대로다.
3. 감독기 진입 전에 contract typed cumulative-before/limit, original_calls, commands/argv/wall/RSS, source hash, output freshness를 확인한다. Immutable E267 자체는 변경하지 않는다. E345의 별도 orchestration wrapper는 actual process/PID creation times를 시작 전부터 관찰하고, supervisor report 누락·실패에서도 첫 오류와 알려진 PID identities/경과 비용을 저장한다. 누락을 성공/0 비용으로 바꾸지 않는다. unowned kill 금지. 정상 경로는 E267의 실제 보고서와 identity/cumulative 링크를 대조한다.
4. 생성된 small contract를 써서 누적 필드 누락을 child spawn 이전 거부하고, 정상 원본-equivalent supervisor-contract 형식이 실제 supervisor 보고서를 내는지 CPU-only로 확인한다. real original payload의 hash/header/read는 없다. metadata whitelist와 모든 binding coverage 검증은 original source-path 분기 자체에 넣어 다시 같은 count 오류가 GPU 단계까지 가는 것을 막는다.
5. 원본 출력/계약 이름은 E345로 구분하며 E344 evidence는 보존한다. 변경된 source SHA의 CPU/native20/50/同 audit 자격과 passive freeze를 새로 닫은 뒤, 같은 후보의 original1.5s body80 + independent audit12를 한번만 실행한다. Root source approval가 마지막 실행 조건이다. 원본 scientific failure는 정상 판정으로 기록한다. 결과가 bout/cycles/slip 등 하나라도 실패하면10s/채택/F1–F8로 승격하지 않는다.

## 자원

E344 기존 original cumulative9616.333529999503 + unknown supervisor body reservation80 = **9696.333529999503s**. 기존 ROOT stage allocation9780, reserved0, jobs0. 사용자 총 시간/토큰 예산은 없다. Active plan의 local singleGPU experiment per-run300wall/31.5model seconds를 유지한다. E345 prospective stage allocation **9900s**, body80+audit12, retry0. 원본 예약은 current qualification 후 적용한다. E344 실패 비용을 지우지 않는다.

E345 Sol FIRST+ONE R1 generated cap20s/32MiB, ROOT current generated qualification cap45s/32MiB. Original scientific internal60/final8/internal72/supervisor80, audit internal10/supervisor12, same native CPU full trace301/2MiB, stage32MiB/raw16MiB/failure4MiB. Oversized/latefail revoke. Generated qualification is engineering only. Current candidates, previous failures, M2/UI/F6 boundaries and biological criteria remain unchanged.

## 구현 범위

Existing common-descent body runner/test may be minimally refactored to use verified E345 registration/complete135-input metadata and new E345 output/contract/freeze paths, with no numerical/physics math changes. New small orchestration/source-qualified registration helpers belong to E345. ROOT controls research docs/checkpoint and actual original reservations; Sol cannot run original arrays/NN/body or expand budgets. Astra review reads bounded source/JSON metadata only; numeric tests/NPZ/NPY/source original payload hashing0. The original factory is still unverified.
